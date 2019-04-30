#include "cricket-checkpoint.h"

#include "defs.h"
#include "cudadebugger.h"
#include "cuda-api.h"

#include "inferior.h"
#include "top.h"
#include "main.h"

#include <stdio.h>
#include <string.h>
#include <sys/time.h>

#include "cricket-cr.h"
#include "cricket-device.h"
#include "cricket-stack.h"
#include "cricket-register.h"
#include "cricket-file.h"
#include "cricket-heap.h"
#include "cricket-elf.h"
#include "cricket-utils.h"
#include "cricket-types.h"

int cricket_checkpoint(int argc, char *argv[])
{
    // TODO: Make this a parameter
    char *ckp_dir = "/home/nei/tmp/cricket-ckp";
    // char *ckp_dir = "/global/work/share/ckp";
    const char *kernel_name = NULL;
    const char *warp_kn;
    cricket_callstack callstack;
    cricket_function_info *function_info = NULL;
    int ret = -1;
    size_t fi_num;
    uint32_t calldepth;
    uint32_t first_warp = 0;
    uint32_t numDev = 0;
    uint64_t *warp_mask;
    uint64_t relative_ssy;
    CUDBGResult res;
    CUDBGAPI cudbgAPI;
#ifdef CRICKET_PROFILE
    struct timeval a, b, c, d, e, f;
    struct timeval la, lb, lc, ld, le, lf, lg;
    gettimeofday(&a, NULL);
#endif

    if (argc != 3) {
        printf("wrong number of arguments, use: %s <pid>\n", argv[0]);
        return -1;
    }

    cricket_init_gdb(argv[0]);

    /* attach to process (both CPU and GPU) */
    printf("attaching...\n");
    attach_command(argv[2], !batch_flag);

    if (cuda_api_get_state() != CUDA_API_STATE_INITIALIZED) {
        printf("Cuda api not initialized!\n");
        return -1;
    } else if (cuda_api_get_attach_state() != CUDA_ATTACH_STATE_COMPLETE) {
        printf("Cuda api not attached!\n");
        return -1;
    } else {
        printf("Cuda api initialized and attached!\n");
    }
#ifdef CRICKET_PROFILE
    gettimeofday(&b, NULL);
#endif

    /* get CUDA debugger API */
    res = cudbgGetAPI(CUDBG_API_VERSION_MAJOR, CUDBG_API_VERSION_MINOR,
                      CUDBG_API_VERSION_REVISION, &cudbgAPI);
    if (res != CUDBG_SUCCESS) {
        printf("%d:", __LINE__);
        goto cuda_error;
    }
    printf("got API\n");

    if (!cricket_device_get_num(cudbgAPI, &numDev)) {
        printf("error getting device num\n");
        goto detach;
    } else if (numDev != 1) {
        printf("expected exactly one CUDA device. Found %u\n", numDev);
        goto detach;
    }

    CricketDeviceProp dev_prop;
    if (!cricket_device_get_prop(cudbgAPI, 0, &dev_prop)) {
        printf("error getting device properties\n");
        goto detach;
    }
    cricket_device_print_prop(&dev_prop);

    warp_mask = malloc(sizeof(uint64_t) * dev_prop.numSMs);

    res = cudbgAPI->readValidWarps(0, 0, warp_mask);
    if (res != CUDBG_SUCCESS) {
        printf("%d:", __LINE__);
        goto cuda_error;
    }
    for (first_warp = 0; first_warp != dev_prop.numWarps; first_warp++) {
        if (*warp_mask & (1LU << first_warp)) {
            if (cricket_cr_kernel_name(cudbgAPI, 0, 0, first_warp,
                                       &kernel_name)) {
                break;
            }
        }
    }
    if (kernel_name == NULL) {
        fprintf(stderr, "cricket-checkpoint: error getting kernel name!\n");
        goto detach;
    } else {
        printf("checkpointing kernel with name: \"%s\"\n", kernel_name);
    }

    if (!cricket_elf_build_fun_info(&function_info, &fi_num)) {
        fprintf(stderr, "failed to build function info array\n");
        goto detach;
    }
    /* for (int i = 0; i < fi_num; ++i) {
         printf("name: %s, has_room: %d, room %lu\n", function_info[i].name,
     function_info[i].has_room, function_info[i].room);
     }*/

    cricket_elf_info elf_info;
    cricket_elf_get_info(kernel_name, &elf_info);
    printf("stack-size: %u, param-addr: %u, param-size: %u, param-num: %lu\n",
           elf_info.stack_size, elf_info.param_addr, elf_info.param_size,
           elf_info.param_num);

    cricketWarpInfo warp_info = { 0 };
    warp_info.dev = 0;
    warp_info.dev_prop = &dev_prop;
    warp_info.sm = 0;
    warp_info.warp = 0;

#ifdef CRICKET_PROFILE
    gettimeofday(&d, NULL);
#endif

    for (int sm = 0; sm != dev_prop.numSMs; sm++) {
        res = cudbgAPI->readValidWarps(0, sm, warp_mask + sm);
        if (res != CUDBG_SUCCESS) {
            printf("%d:", __LINE__);
            goto cuda_error;
        }
        printf("SM %u: %lx - ", sm, warp_mask[sm]);
        print_binary64(warp_mask[sm]);

        for (uint8_t warp = 0; warp != dev_prop.numWarps; warp++) {
            if (warp_mask[sm] & (1LU << warp)) {
#ifdef CRICKET_PROFILE
                gettimeofday(&la, NULL);
#endif
                if (!cricket_cr_kernel_name(cudbgAPI, 0, sm, warp, &warp_kn)) {
                    fprintf(stderr, "cricket-checkpoint: could not get kernel "
                                    "name for D%uS%uW%u\n",
                            0, sm, warp);
                    goto detach;
                }
                if (strcmp(warp_kn, kernel_name) != 0) {
                    // TODO: here are some arguments missing
                    fprintf(stderr,
                            "cricket-checkpoint: found kernel \"%s\" while "
                            "checkpointing kernel \"%s\". only one kernel can "
                            "be checkpointed\n");
                    goto detach;
                }

                warp_info.sm = sm;
                warp_info.warp = warp;
                warp_info.kernel_name = kernel_name;
                warp_info.stack_size = elf_info.stack_size;

                if (!cricket_cr_callstack(cudbgAPI, &warp_info,
                                          CRICKET_CR_NOLANE, &callstack)) {
                    fprintf(stderr, "failed to get callstack\n");
                    goto detach;
                }

                printf("SM %u warp %u (active): %x - ", sm, warp,
                       callstack.active_lanes);
                print_binary32(callstack.active_lanes);
                printf("SM %u warp %u (valid): %x - ", sm, warp,
                       callstack.valid_lanes);
                print_binary32(callstack.valid_lanes);

#ifdef CRICKET_PROFILE
                gettimeofday(&lc, NULL);
#endif
                if (!cricket_cr_make_checkpointable(cudbgAPI, &warp_info,
                                                    function_info, fi_num,
                                                    &callstack)) {
                    fprintf(stderr, "cricket-checkpoint: could not make "
                                    "checkpointable.\n");
                    goto detach;
                }
#ifdef CRICKET_PROFILE
                gettimeofday(&ld, NULL);
#endif

                if (!cricket_cr_ckp_pc(cudbgAPI, &warp_info, CRICKET_CR_NOLANE,
                                       ckp_dir, &callstack)) {
                    fprintf(stderr, "cricket-checkpoint: ckp_pc failed\n");
                    goto detach;
                }
                cricket_cr_free_callstack(&callstack);
#ifdef CRICKET_PROFILE
                gettimeofday(&le, NULL);
#endif
                for (uint8_t lane = 0; lane != dev_prop.numLanes; lane++) {
                    if (callstack.valid_lanes & (1LU << lane)) {
                        cricket_cr_ckp_lane(cudbgAPI, &warp_info, lane,
                                            ckp_dir);
                    }
                }
#ifdef CRICKET_PROFILE
                gettimeofday(&lf, NULL);
#endif

                if (!cricket_cr_ckp_shared(cudbgAPI, ckp_dir, &elf_info, 0, sm,
                                           warp)) {
                    printf("cricket_cr_ckp_shared unsuccessful\n");
                }
#ifdef CRICKET_PROFILE
                gettimeofday(&lg, NULL);
                double lct = ((double)((lc.tv_sec * 1000000 + lc.tv_usec) -
                                       (la.tv_sec * 1000000 + la.tv_usec))) /
                             1000000.;
                double ldt = ((double)((ld.tv_sec * 1000000 + ld.tv_usec) -
                                       (lc.tv_sec * 1000000 + lc.tv_usec))) /
                             1000000.;
                double let = ((double)((le.tv_sec * 1000000 + le.tv_usec) -
                                       (ld.tv_sec * 1000000 + ld.tv_usec))) /
                             1000000.;
                double lft = ((double)((lf.tv_sec * 1000000 + lf.tv_usec) -
                                       (le.tv_sec * 1000000 + le.tv_usec))) /
                             1000000.;
                double lgt = ((double)((lg.tv_sec * 1000000 + lg.tv_usec) -
                                       (lf.tv_sec * 1000000 + lf.tv_usec))) /
                             1000000.;
                printf("warp time:\n\tPROFILE misc: %f s\n\tPROFILE "
                       "checkpointable: %f "
                       "s\n\tPROFILE pc: %f s\n\tPROFILE lane: %f s\n\tPROFILE "
                       "shared: "
                       "%f s\n",
                       lct, ldt, let, lft, lgt);
#endif
            }
        }
    }

#ifdef CRICKET_PROFILE
    gettimeofday(&e, NULL);
#endif

    if (!cricket_cr_ckp_params(cudbgAPI, ckp_dir, &elf_info, 0, 0,
                               first_warp)) {
        printf("cricket_cr_ckp_params unsuccessful\n");
    }

    if (!cricket_cr_ckp_globals(cudbgAPI, ckp_dir)) {
        printf("cricket_cr_ckp_globals unsuccessful\n");
    }

#ifdef CRICKET_PROFILE
    gettimeofday(&f, NULL);

    double bt = ((double)((b.tv_sec * 1000000 + b.tv_usec) -
                          (a.tv_sec * 1000000 + a.tv_usec))) /
                1000000.;
    double dt = ((double)((d.tv_sec * 1000000 + d.tv_usec) -
                          (b.tv_sec * 1000000 + b.tv_usec))) /
                1000000.;
    double et = ((double)((e.tv_sec * 1000000 + e.tv_usec) -
                          (d.tv_sec * 1000000 + d.tv_usec))) /
                1000000.;
    double ft = ((double)((f.tv_sec * 1000000 + f.tv_usec) -
                          (e.tv_sec * 1000000 + e.tv_usec))) /
                1000000.;
    double comt = ((double)((f.tv_sec * 1000000 + f.tv_usec) -
                            (a.tv_sec * 1000000 + a.tv_usec))) /
                  1000000.;
    printf("complete time:\n\tPROFILE attach: %f s\n\tPROFILE init: %f "
           "s\n\tPROFILE inkernel: %f s\n\tPROFILE outkernel: %f s\n\tPROFILE "
           "complete: %f s\n",
           bt, dt, et, ft, comt);
#endif

    ret = 0;
    goto detach;
cuda_error:
    printf("Cuda Error: \"%s\"\n", cudbgGetErrorString(res));
detach:
    free(function_info);
    /* Detach from process (CPU and GPU) */
    detach_command(NULL, !batch_flag);
    /* finalize, i.e. clean up CUDA debugger API */
    cuda_api_finalize();

    /* quit GDB. TODO: Why is this necccessary? */
    quit_force(NULL, 0);
    return ret;
}

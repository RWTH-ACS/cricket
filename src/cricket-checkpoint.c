#include "cricket-checkpoint.h"

#include "cuda-api.h"
#include "cudadebugger.h"
#include "defs.h"

#include "inferior.h"
#include "main.h"
#include "top.h"

#include <stdio.h>
#include <string.h>
#include <sys/time.h>

#include "cricket-cr.h"
#include "cricket-device.h"
#include "cricket-elf.h"
#include "cricket-file.h"
#include "cricket-heap.h"
#include "cricket-register.h"
#include "cricket-stack.h"
#include "cricket-types.h"
#include "cricket-utils.h"

struct profile_times_warp
{
    struct timeval misc_begin, checkpointable_begin, pc_begin, shared_begin,
        lanes_begin, warp_end;
    uint warp_id, sm_id;
};

struct profile_times
{
    struct timeval attach_begin, init_begin, inkernel_begin, outkernel_begin,
        checkpoint_end;
    struct profile_times_warp *warp;
    uint entry_count, nr_sm, nr_warps;
};

void print_profile_times(const struct profile_times *pt)
{
    double attach_time = time_diff_sec(&pt->init_begin, &pt->attach_begin);
    double init_time = time_diff_sec(&pt->inkernel_begin, &pt->init_begin);
    double inkernel_time =
        time_diff_sec(&pt->outkernel_begin, &pt->inkernel_begin);
    double outkernel_time =
        time_diff_sec(&pt->checkpoint_end, &pt->outkernel_begin);
    double complete_time =
        time_diff_sec(&pt->checkpoint_end, &pt->attach_begin);

    printf("\n============= Profiling Results =============\n");
    printf("Total duration:\n");
    printf("\tattach:    %f s\n"
           "\tinit:      %f s\n"
           "\tinkernel:  %f s\n"
           "\toutkernel: %f s\n"
           "\tcomplete:  %f s\n\n",
           attach_time, init_time, inkernel_time, outkernel_time,
           complete_time);

    for (uint i = 0; i < pt->entry_count; ++i) {
        uint misc_time = time_diff_usec(&pt->warp[i].checkpointable_begin,
                                        &pt->warp[i].misc_begin);
        uint checkpointable_time = time_diff_usec(
            &pt->warp[i].pc_begin, &pt->warp[i].checkpointable_begin);
        uint pc_time =
            time_diff_usec(&pt->warp[i].lanes_begin, &pt->warp[i].pc_begin);
        uint lane_time =
            time_diff_usec(&pt->warp[i].shared_begin, &pt->warp[i].lanes_begin);
        uint shared_time =
            time_diff_usec(&pt->warp[i].warp_end, &pt->warp[i].shared_begin);
        uint warp_total =
            time_diff_usec(&pt->warp[i].warp_end, &pt->warp[i].misc_begin);

        printf("SM %d Warp %d duration:\n", pt->warp[i].sm_id,
               pt->warp[i].warp_id);
        printf("\tmisc:           %u us\n"
               "\tcheckpointable: %u us\n"
               "\tpc:             %u us\n"
               "\tlane:           %u us\n"
               "\tshared:         %u us\n"
               "\twarp:           %u us\n\n",
               misc_time, checkpointable_time, pc_time, lane_time, shared_time,
               warp_total);
    }
}

int cricket_checkpoint(char *pid, const char *ckp_dir, const int profile)
{
    char *kernel_name = NULL;
    char *warp_kn;
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

    struct profile_times timestamps;
    if (profile) {
        timestamps.entry_count = 0;
        gettimeofday(&timestamps.attach_begin, NULL);
    }

    /* attach to process (both CPU and GPU) */
    printf("attaching to PID %s\n", pid);
    // TODO check if parameter is a valid pid
    attach_command(pid, !batch_flag);

    if (cuda_api_get_state() != CUDA_API_STATE_INITIALIZED) {
        printf("Cuda api not initialized!\n");
        return -1;
    } else if (cuda_api_get_attach_state() != CUDA_ATTACH_STATE_COMPLETE) {
        printf("Cuda api not attached!\n");
        return -1;
    } else {
        printf("Cuda api initialized and attached!\n");
    }

    if (profile) {
        gettimeofday(&timestamps.init_begin, NULL);
    }

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

    // checkpointing kernel
    if (profile) {
        gettimeofday(&timestamps.inkernel_begin, NULL);
    }


    if (profile) {
        timestamps.nr_warps = dev_prop.numWarps;
        timestamps.nr_sm = dev_prop.numSMs;
        printf("======== nr warps %d ===\n", dev_prop.numWarps);
        printf("======== nr sm  %d ===\n", dev_prop.numSMs);
        timestamps.warp = malloc(timestamps.nr_sm * timestamps.nr_warps *
                                 sizeof(struct profile_times_warp));
    }

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
                if (profile) {
                    timestamps.warp[timestamps.entry_count].warp_id = warp,
                    timestamps.warp[timestamps.entry_count].sm_id = sm,
                    gettimeofday(
                        &(timestamps.warp[timestamps.entry_count].misc_begin),
                        NULL);
                }

                if (!cricket_cr_kernel_name(cudbgAPI, 0, sm, warp, &warp_kn)) {
                    fprintf(stderr, "cricket-checkpoint: could not get kernel "
                                    "name for D%uS%uW%u\n",
                            0, sm, warp);
                    goto detach;
                }
                if (strcmp(warp_kn, kernel_name) != 0) {
                    // TODO: here are some arguments missing
                    fprintf(stderr, "cricket-checkpoint: found kernel \"%s\" "
                                    "while checkpointing kernel \"%s\". only "
                                    "one "
                                    "kernel can be checkpointed\n");
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

                if (profile) {
                    gettimeofday(&(timestamps.warp[timestamps.entry_count]
                                       .checkpointable_begin),
                                 NULL);
                    printf("=== Debug timestamp %d ",
                           timestamps.warp[timestamps.entry_count]
                               .checkpointable_begin);
                }

                if (!cricket_cr_make_checkpointable(cudbgAPI, &warp_info,
                                                    function_info, fi_num,
                                                    &callstack)) {
                    fprintf(stderr, "cricket-checkpoint: could not make "
                                    "checkpointable.\n");
                    goto detach;
                }

                if (profile) {
                    gettimeofday(
                        &(timestamps.warp[timestamps.entry_count].pc_begin),
                        NULL);
                }
                if (!cricket_cr_ckp_pc(cudbgAPI, &warp_info, CRICKET_CR_NOLANE,
                                       ckp_dir, &callstack)) {
                    fprintf(stderr, "cricket-checkpoint: ckp_pc failed\n");
                    goto detach;
                }
                cricket_cr_free_callstack(&callstack);

                if (profile) {
                    gettimeofday(
                        &(timestamps.warp[timestamps.entry_count].lanes_begin),
                        NULL);
                }
                for (uint8_t lane = 0; lane != dev_prop.numLanes; lane++) {
                    if (callstack.valid_lanes & (1LU << lane)) {
                        cricket_cr_ckp_lane(cudbgAPI, &warp_info, lane,
                                            ckp_dir);
                    }
                }

                if (profile) {
                    gettimeofday(
                        &(timestamps.warp[timestamps.entry_count].shared_begin),
                        NULL);
                }

                if (!cricket_cr_ckp_shared(cudbgAPI, ckp_dir, &elf_info, 0, sm,
                                           warp)) {
                    printf("cricket_cr_ckp_shared unsuccessful\n");
                }

                if (profile) {
                    gettimeofday(
                        &(timestamps.warp[timestamps.entry_count].warp_end),
                        NULL);
                    timestamps.entry_count += 1;
                }
            }
        }
    }

    // Checkpoint non-kernel data
    if (profile) {
        gettimeofday(&timestamps.outkernel_begin, NULL);
    }

    if (!cricket_cr_ckp_params(cudbgAPI, ckp_dir, &elf_info, 0, 0,
                               first_warp)) {
        printf("cricket_cr_ckp_params unsuccessful\n");
    }

    if (!cricket_cr_ckp_globals(cudbgAPI, ckp_dir)) {
        printf("cricket_cr_ckp_globals unsuccessful\n");
    }

    if (profile) {
        gettimeofday(&timestamps.checkpoint_end, NULL);
        print_profile_times(&timestamps);
    }

    ret = 0;
    goto detach;
cuda_error:
    printf("Cuda Error: \"%s\"\n", cudbgGetErrorString(res));
detach:
    free(function_info);
    if (profile) {
        free(timestamps.warp);
    }
    /* Detach from process (CPU and GPU) */
    detach_command(NULL, !batch_flag);
    /* finalize, i.e. clean up CUDA debugger API */
    cuda_api_finalize();

    /* quit GDB. TODO: Why is this necccessary? */
    quit_force(NULL, 0);
    return ret;
}

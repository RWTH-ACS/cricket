#include "../gdb/config.h"
#include "defs.h"
#include <stdio.h>
#include "cudadebugger.h"
#include "cuda-api.h"
#include "cuda-options.h"
#include "inferior.h"
#include "top.h"
#include "bfd.h"
#include "cuda-tdep.h"
#include "objfiles.h"
#include "cuda-state.h"
#include "gdbcore.h"
#include "regcache.h"
#include "cli/cli-cmds.h"
#include "cli/cli-setshow.h"
#include "interps.h"
#include "main.h"
#include <dlfcn.h>
#include <sys/time.h>
#include "signals-state-save-restore.h"

#include "cricket-util.h"
#include "cricket-cr.h"
#include "cricket-device.h"
#include "cricket-stack.h"
#include "cricket-register.h"
#include "cricket-file.h"
#include "cricket-heap.h"
#include "cricket-elf.h"

#define CRICKET_PROFILE 1

CUDBGAPI cudbgAPI = NULL;

bool cricket_print_lane_states(CUDBGAPI cudbgAPI, CricketDeviceProp *dev_prop)
{
    uint64_t warp_mask;
    uint64_t warp_mask_broken;
    uint32_t lanemask;
    CUDBGResult res;
    for (int sm = 0; sm != dev_prop->numSMs; sm++) {
        res = cudbgAPI->readValidWarps(0, sm, &warp_mask);
        if (res != CUDBG_SUCCESS) {
            printf("%d:", __LINE__);
            goto cuda_error;
        }
        if (warp_mask > 0) {
            printf("SM %u: %lx - ", sm, warp_mask);
            print_binary64(warp_mask);
        }
        res = cudbgAPI->readBrokenWarps(0, sm, &warp_mask_broken);
        if (res != CUDBG_SUCCESS) {
            printf("%d:", __LINE__);
            goto cuda_error;
        }
        if (warp_mask > 0) {
            printf("broken:%s- ", (sm > 10 ? "  " : " "));
            print_binary64(warp_mask_broken);
        }

        for (uint8_t warp = 0; warp != dev_prop->numWarps; warp++) {
            if (warp_mask & (1LU << warp)) {
                res = cudbgAPI->readValidLanes(0, sm, warp, &lanemask);
                if (res != CUDBG_SUCCESS) {
                    printf("%d:", __LINE__);
                    goto cuda_error;
                }
                printf("warp %u (valid): %x - ", warp, lanemask);
                print_binary32(lanemask);
                res = cudbgAPI->readActiveLanes(0, sm, warp, &lanemask);
                if (res != CUDBG_SUCCESS) {
                    printf("%d:", __LINE__);
                    goto cuda_error;
                }
                printf("warp %u (active): %x - ", warp, lanemask);
                print_binary32(lanemask);
            }
        }
    }
    return true;
cuda_error:
    printf("Cuda Error: \"%s\"\n", cudbgGetErrorString(res));
    return false;
}

bool cricket_all_warps_broken(CUDBGAPI cudbgAPI, CricketDeviceProp *dev_prop)
{
    uint64_t warp_mask;
    uint64_t warp_mask_broken;
    CUDBGResult res;
    for (int sm = 0; sm != dev_prop->numSMs; sm++) {
        res = cudbgAPI->readValidWarps(0, sm, &warp_mask);
        if (res != CUDBG_SUCCESS) {
            fprintf(stderr, "%d:", __LINE__);
            goto cuda_error;
        }
        res = cudbgAPI->readBrokenWarps(0, sm, &warp_mask_broken);
        if (res != CUDBG_SUCCESS) {
            fprintf(stderr, "%d:", __LINE__);
            goto cuda_error;
        }
        if (warp_mask != warp_mask_broken)
            return false;
    }
    return true;
cuda_error:
    fprintf(stderr, "Cuda Error: \"%s\"\n", cudbgGetErrorString(res));
    return false;
}

extern char *gdb_program_name;

bool cricket_init_gdb(char *name)
{
    lim_at_start = (char *) sbrk (0);
    /* initialize gdb streams, necessary for gdb_init */
    notice_open_fds();
    saved_command_line = (char*) xstrdup("");
    main_ui = new ui(stdin, stdout, stderr);
    current_ui = main_ui;
    gdb_stdtargerr = gdb_stderr;
    gdb_stdtargin = gdb_stdin;

    /* initialize BFD, the binary file descriptor library */
    if (bfd_init() != BFD_INIT_MAGIC) {
        printf("error bfd init\n");
        return false;
    }

    gdb_program_name = xstrdup (name);
    current_directory = getcwd (NULL, 0);
    if (current_directory == NULL)
        printf("error finding working directory\n");

    /* initialize gdb paths */
    gdb_sysroot = relocate_gdb_directory(TARGET_SYSTEM_ROOT,
                                         TARGET_SYSTEM_ROOT_RELOCATABLE);
    if (gdb_sysroot == NULL || *gdb_sysroot == '\0') {
        xfree(gdb_sysroot);
        gdb_sysroot = xstrdup (TARGET_SYSROOT_PREFIX);
    }
    debug_file_directory = relocate_gdb_directory (DEBUGDIR, DEBUGDIR_RELOCATABLE);
    gdb_datadir = relocate_gdb_directory (GDB_DATADIR, GDB_DATADIR_RELOCATABLE);

    /* tell gdb that we do not want to run an interactive shell */
    batch_flag = 1;


    save_original_signals_state(batch_flag);
    /* initialize GDB */
    printf("gdb_init...\n");
    gdb_init(gdb_program_name);

    char *interpreter_p = xstrdup(INTERP_CONSOLE);
    set_top_level_interpreter(interpreter_p);
    return true;
}

int cricket_analyze(int argc, char *argv[])
{
    if (argc != 3) {
        printf("wrong number of arguments, use: %s <executable>\n", argv[0]);
        return -1;
    }
    printf("Analyzing \"%s\"\n", argv[2]);
    if (!cricket_elf_analyze(argv[2])) {
        printf("cricket analyze unsuccessful\n");
        return -1;
    }
    return 0;
}

int cricket_info(int argc, char *argv[])
{
    bfd *hostbfd = NULL;
    bfd *cudabfd = NULL;
    asection *section;
    asection *cudasection;
    size_t fatbin_pos, fatbin_size;
    FILE *hostbfd_fd = NULL;
    FILE *cudabfd_fd = NULL;
    void *fatbin = NULL;
    int ret = -1;
    size_t prefixlen = strlen(CRICKET_ELF_NV_INFO_PREFIX);
    char *filename = NULL;

    cricket_elf_info info = {0};
    char data[8];
    uint32_t kernel_index;
    size_t attr_num;
    char *attrs;

    size_t symtab_size;
    asymbol **symtab;
    size_t symtab_num;
    size_t i;

    if (argc != 3) {
        fprintf(stderr, "wrong number of arguments, use: %s <executable>\n", argv[0]);
        return -1;
    }
    filename = argv[2];

    if (filename == NULL) {
        fprintf(stderr, "cricket_elf (%d): filename is NULL\n", __LINE__);
        return -1;
    }

    bfd_init();

    if ((hostbfd_fd = fopen(filename, "rb")) == NULL) {
        fprintf(stderr, "cricket (%d): fopen failed\n", __LINE__);
        return -1;
    }

    if ((hostbfd = bfd_openstreamr(filename, NULL, hostbfd_fd)) == NULL) {
        fprintf(stderr, "cricket (%d): bfd_openr failed on %s\n", __LINE__,
                filename);
        fclose(hostbfd_fd);
        goto cleanup;
    }

    if (!bfd_check_format(hostbfd, bfd_object)) {
        fprintf(stderr, "cricket (%d): %s has wrong bfd format\n", __LINE__,
                filename);
        goto cleanup;
    }

    for (section = hostbfd->sections; section != NULL; section = section->next) {
        if (strcmp(section->name, CRICKET_ELF_FATBIN) != 0) {
            continue;
        } else {
            //There is just one fatbin section so stop searching...
            break;
        }
    }
    //printf("found fatbin @ %p, size: %x\n", section->filepos, section->size);
    if ((fatbin = malloc(section->size)) == NULL) {
        goto cleanup;
    }
    if (fseek(hostbfd_fd, section->filepos, SEEK_SET) != 0) {
        fprintf(stderr, "cricket: fseek failed\n");
        goto cleanup;
    }
    if (fread(fatbin, section->size, 1, hostbfd_fd) != 1) {
        fprintf(stderr, "cricket: fread failed\n");
        goto cleanup;
    }

    //Now read CUBIN offsets from symtab

    if ((symtab_size = bfd_get_symtab_upper_bound(hostbfd)) < 0) {
        fprintf(stderr, "cricket (%d): getting symtab size failed\n", __LINE__);
        goto cleanup;
    }

    if ((symtab = (asymbol**)malloc(symtab_size)) == NULL) {
        fprintf(stderr, "cricket (%d): malloc for symtab failed\n", __LINE__);
        goto cleanup;
    }

    if ((symtab_num = bfd_canonicalize_symtab(hostbfd, symtab)) < 0) {
        fprintf(stderr, "cricket (%d): canonicalize symtab failed\n", __LINE__);
        goto cleanup;
    }

    for (i = 0; i < symtab_num; i++) {
        if (!symtab[i] || strcmp(symtab[i]->name, "fatbinData") != 0) {
            continue;
        }
        //printf("Found CUBIN at offset 0x%x\n", symtab[i]->value);



        /* There is a 0x50 byte header in front of the actual ELF. I don't know what
         * information the header contains. 
         * Stackoverflow suggests the following (I'm not convinced...)
         * struct {
         *     uint32_t magic; // Always 0x466243b1
         *     uint32_t seq;   // Sequence number of the cubin
         *     uint64_t ptr;   // The pointer to the real cubin
         *     uint64_t data_ptr;    // Some pointer related to the data segment
         * }
         * https://stackoverflow.com/questions/6392407/what-are-the-parameters-for-cudaregisterfatbinary-and-cudaregisterfunction-f/39453201#39453201
         */
        fatbin_pos = symtab[i]->value + 0x50;
        fatbin_size = section->size - symtab[i]->value - 0x50;

        if ((cudabfd_fd = fmemopen(fatbin+fatbin_pos, fatbin_size, "rb")) == NULL) {
            fprintf(stderr, "cricket (%d): fmemopen failed\n", __LINE__);
            goto cleanup;
        }

        if ((cudabfd = bfd_openstreamr(filename, NULL, cudabfd_fd)) == NULL) {
            fprintf(stderr, "cricket: bfd_openstreamr failed\n");
            fclose(cudabfd_fd);
            goto cleanup;
        }

        if (!bfd_check_format(cudabfd, bfd_object)) {
            fprintf(stderr, "cricket: wrong bfd format\n");
            goto cleanup;
        }


        for (cudasection = cudabfd->sections; cudasection != NULL;
             cudasection = cudasection->next) {

            if (strncmp(cudasection->name, CRICKET_ELF_NV_INFO_PREFIX,
                        prefixlen + 1) == 0) {
                if (!cricket_elf_extract_attribute(
                         cudabfd, cudasection, EIATTR_MIN_STACK_SIZE, 8, data,
                         NULL, NULL)) {
                    fprintf(stderr,
                            "error: found .nv.info section but could not find "
                            "stack size for kernel %s\n",
                            cudasection->name);
                }
                info.stack_size = *(uint32_t *)(data + 4);
            } else if (strncmp(cudasection->name, CRICKET_ELF_NV_INFO_PREFIX,
                               prefixlen) == 0) {
                if (!cricket_elf_extract_multiple_attributes(
                         cudabfd, cudasection, EIATTR_KPARAM_INFO, 12, (void **)&attrs,
                         &attr_num)) {
                    fprintf(stderr, "warning: found %s section but "
                                    "could not find any EIATTR_KPARAM_INFO attributes. "
                                    "Maybe the kernel has no parameters?\n",
                            cudasection->name);
                    attr_num = 0;
                    attrs = NULL;
                }
                info.params = malloc(attr_num * sizeof(cricket_param_info));
                info.param_num = attr_num;
                printf("%s param_num %d\n", cudasection->name+prefixlen+1,
                       info.param_num);
                for (int i = 0; i != attr_num; ++i) {
                    info.params[i].index = *(uint16_t *)(attrs + 4 + i * 12);
                    info.params[i].offset = *(uint16_t *)(attrs + 6 + i * 12);
                    info.params[i].size = *(uint8_t *)(attrs + 10 + i * 12) >> 2;
                }
                printf("%s param_offsets ", cudasection->name+prefixlen+1);
                for (int i = 0; i != attr_num; ++i) {
                    for (int j = 0; j != attr_num; ++j) {
                        if (info.params[j].index == i) {
                            printf("%04x,", info.params[j].offset);
                            break;
                        }
                    }
                }
                printf("\n");
                printf("%s param_sizes ", cudasection->name+prefixlen+1);
                for (int i = 0; i != attr_num; ++i) {
                    for (int j = 0; j != attr_num; ++j) {
                        if (info.params[j].index == i) {
                            printf("%04x,", info.params[j].size);
                            break;
                        }
                    }
                }
                printf("\n");
                free(attrs);
                if (!cricket_elf_extract_attribute(cudabfd, cudasection,
                                                   EIATTR_PARAM_CBANK, 8, data,
                                                   NULL, NULL)) {
                    fprintf(stderr, "warning: found %s section but "
                                    "could not find EIATTR_PARAM_CBANK attribute. "
                                    "Maybe the kernel has no parameters?\n",
                            cudasection->name);
                   memset(data, 0, 8);
                }
                info.param_size = *(uint16_t *)(data + 6);
                info.param_addr = *(uint16_t *)(data + 4);
                printf("%s param_size %d\n", cudasection->name+prefixlen+1,
                       info.param_size);
                printf("%s param_addr %d\n", cudasection->name+prefixlen+1,
                       info.param_addr);
            } else if (strncmp(cudasection->name, CRICKET_ELF_NV_SHARED_PREFIX,
                               strlen(CRICKET_ELF_NV_SHARED_PREFIX) - 1) == 0) {

                if (!cricket_elf_extract_shared_size(cudasection,
                                                     &info.shared_size)) {
                    fprintf(stderr, "warning: could not read shared memory size\n");
                }
                printf("%s shared %d\n", cudasection->name+strlen(CRICKET_ELF_NV_SHARED_PREFIX),
                       info.shared_size);
            }

        }
        if (cudabfd != NULL) {
            bfd_close(cudabfd);
            cudabfd = NULL;
        }
    }

    ret = 0;
cleanup:
    free(fatbin);
    if (cudabfd != NULL)
        bfd_close(cudabfd);
    if (hostbfd != NULL)
        bfd_close(hostbfd);
    return ret;
}

static void
symbol_file_add_main_adapter (const char *arg, int from_tty)
{
  symfile_add_flags add_flags = 0;

  if (from_tty)
    add_flags |= SYMFILE_VERBOSE;

  symbol_file_add_main (arg, add_flags);
}

int cricket_restore(int argc, char *argv[])
{
    CUDBGResult res;
    CUDBGAPI cudbgAPI;
    cricketWarpInfo warp_info = { 0 };
    cricket_callstack callstack;
    char *patched_binary = "/tmp/cricket-ckp/patched_binary";
    const char *ckp_dir = "/tmp/cricket-ckp";
    const char *kernel_name;
    uint32_t active_lanes;
    uint32_t first_warp;
    cricket_jmptable_index *jmptbl;
    uint64_t warp_mask;
    size_t jmptbl_len;
    if (argc != 3) {
        printf("wrong number of arguments, use: %s <executable>\n", argv[0]);
        return -1;
    }

#ifdef CRICKET_PROFILE
    // a-b = PROFILE patch
    struct timeval a, b, c, d, e, f, g;
    gettimeofday(&a, NULL);
#endif

    /* Patches binary, replacing breakpoints in all code segments
     * with the jumptable that is able to restore any
     * synchornization state (SSY), subcall state (PRET) and
     * PC (JMX/BRX).
     */
    if (!cricket_elf_patch_all(argv[2], patched_binary, &jmptbl, &jmptbl_len)) {
        fprintf(stderr, "cricket-cr: error while patching binary\n");
        return -1;
    }

    // Print the jumptable
    for (size_t i = 0; i < jmptbl_len; ++i) {
        printf("\t\"%s\"\n", jmptbl[i].function_name);
        for (size_t j = 0; j < jmptbl[i].ssy_num; ++j) {
            printf("\t\tSSY@%lx->%lx\n", jmptbl[i].ssy[j].address,
                   jmptbl[i].ssy[j].destination);
        }
        for (size_t j = 0; j < jmptbl[i].cal_num; ++j) {
            printf("\t\tPRET@%lx->%lx\n", jmptbl[i].cal[j].address,
                   jmptbl[i].cal[j].destination);
        }
        printf("\tSYNC@%lx\n\n", jmptbl[i].sync_address);
    }

    /* Read callstacks for all warps from checkpoint file.
     * The highest level function is the kernel we want to start later.
     */
    for (first_warp = 0; first_warp != 32; first_warp++) {
        warp_info.warp = first_warp;
        if (cricket_cr_read_pc(&warp_info, CRICKET_CR_NOLANE, ckp_dir,
                               &callstack)) {
            // fprintf(stderr, "cricket-cr: error while reading pc memory\n");
            break;
        }
    }
    kernel_name = callstack.function_names[callstack.callstack_size - 1];

#ifdef CRICKET_PROFILE
    // b-c = PROFILE runattach
    gettimeofday(&b, NULL);
#endif

    cricket_init_gdb(patched_binary);

    // load the patched binary
    exec_file_attach(patched_binary, !batch_flag);
    symbol_file_add_main_adapter(patched_binary, !batch_flag);

    // break when the kernel launches
    struct cmd_list_element *cl;
    tbreak_command((char *)kernel_name, !batch_flag);

    // launch program until breakpoint is reached
    char *prun = "run";
    cl = lookup_cmd(&prun, cmdlist, "", 0, 1);
    cmd_func(cl, prun, !batch_flag);

    // Use the waiting period to get the CUDA debugger API.
    if (cuda_api_get_state() != CUDA_API_STATE_INITIALIZED) {
        printf("Cuda api not initialized!\n");
        return -1;
        // } else if (cuda_api_get_attach_state() != CUDA_ATTACH_STATE_COMPLETE)
        // {
        //  printf("Cuda api not attached!\n");
        //  return -1;
    } else {
        printf("Cuda api initialized and attached!\n");
    }

    /* get CUDA debugger API */
    res = cudbgGetAPI(CUDBG_API_VERSION_MAJOR, CUDBG_API_VERSION_MINOR,
                      CUDBG_API_VERSION_REVISION, &cudbgAPI);
    if (res != CUDBG_SUCCESS) {
        printf("%d:", __LINE__);
        goto cuda_error;
    }
    printf("cricket: got CUDA debugging API\n");

    // We currently only support a single GPU
    uint32_t numDev = 0;
    if (!cricket_device_get_num(cudbgAPI, &numDev)) {
        printf("error getting device num\n");
        goto detach;
    } else if (numDev != 1) {
        printf("expected exactly one CUDA device. Found %u\n", numDev);
        goto detach;
    }

    // Get device/architecture data so we know how many SMs/warps/lanes
    // to restore
    CricketDeviceProp dev_prop;
    if (!cricket_device_get_prop(cudbgAPI, 0, &dev_prop)) {
        printf("error getting device properties\n");
        goto detach;
    }
    printf("cricket: identified device:\n");
    cricket_device_print_prop(&dev_prop);

    // Wait until all warps have reached the breakpoint at the
    // beginning of the kernel we want to restore
    while (!cricket_all_warps_broken(cudbgAPI, &dev_prop)) {
        printf("waiting for warps to break...\n");
        usleep(500);
    }
    printf("cricket: all warps are now stopped!\n");
#ifdef CRICKET_PROFILE
    // c-d = PROFILE tojmptbl
    gettimeofday(&c, NULL);
#endif
    /* All warps have hit the breakpoint and we can now restore the device state
     */

    // printf("cricket: current lane states:\n");
    // cricket_print_lane_states(cudbgAPI, &dev_prop);

    // cricket_cr_kernel_name(cudbgAPI, 0,0,0, &kernel_name);
    // printf("cricket: kernel-name: \"%s\"\n", kernel_name);

    uint32_t lanemask;
    uint64_t sswarps;
    warp_info.dev = 0;
    warp_info.dev_prop = &dev_prop;
    cricket_jmptable_index *index;
    cricket_jmptable_index *kernelindex;
    uint64_t relative_ssy;
    uint64_t jmptbl_address;
    const char *fn;
    uint32_t predicate = 1;
    uint64_t cur_address = 0;
    uint64_t start_address = 0;
    uint64_t jmptable_addr;
    uint64_t rb_address;
    bool found_callstack = false;

    // We first need to navigate the jumptable in the kernel entry
    // function.
    // Get the jumptable data for this function (each function
    // has its own jumptable).
    if (!cricket_elf_get_jmptable_index(jmptbl, jmptbl_len, kernel_name,
                                        &kernelindex)) {
        fprintf(stderr, "get jmptable entry failed\n");
        goto detach;
    }
    if (kernelindex == NULL) {
        fprintf(stderr, "kernel %s not found\n", kernel_name);
        goto detach;
    }
    // Address where the jumptable starts.
    start_address = kernelindex->start_address + 0x8;
    // The following is required because of control instructions, which
    // are always skipped by the debugger.
    if (start_address % (4 * 8) == 0) {
        start_address += 0x8;
    }
    printf("start_address: %lx, virt: %lx, relative: %lx\n", start_address,
           callstack.pc[callstack.callstack_size - 1].virt,
           callstack.pc[callstack.callstack_size - 1].relative);
    // Read the virtual PC (virtual absolute address) where the warps are
    // currently broken.
    cudbgAPI->readVirtualPC(0, 0, first_warp, 0, &rb_address);
    printf("rb %lx\n", rb_address);
    // Calculate the virtual (absolute) address from current (vitual)
    // address and (relative) jumtable address
    jmptable_addr = rb_address + kernelindex->start_address - 0x8;

    for (int sm = 0; sm != dev_prop.numSMs; sm++) {
        // Only valid warps participate at the kernel's execution
        res = cudbgAPI->readValidWarps(warp_info.dev, sm, &warp_mask);
        if (res != CUDBG_SUCCESS) {
            printf("%d:", __LINE__);
            goto cuda_error;
        }
        if (warp_mask == 0) {
            continue;
        }
        cudbgAPI->readVirtualPC(0, sm, first_warp, 0, &rb_address);
        printf("\trb address: %lx\n", rb_address);
        printf("sm %d: resuming warps %lx until %lx\n", sm, warp_mask,
               jmptable_addr);
        // Goto jumptable with all warps
        res = cudbgAPI->resumeWarpsUntilPC(warp_info.dev, sm, warp_mask,
                                           jmptable_addr);
        if (res != CUDBG_SUCCESS) {
            printf("%d:", __LINE__);
            goto cuda_error;
        }

        // Double check if this worked
        cudbgAPI->readVirtualPC(0, sm, first_warp, 0, &rb_address);
        printf("\trb address: %lx\n", rb_address);

        // Wait until all SMs reached the jumptable
        if (!cricket_cr_sm_broken(cudbgAPI, warp_info.dev, sm)) {
            printf("waiting for sm to break...\n");
            while (!cricket_cr_sm_broken(cudbgAPI, warp_info.dev, sm)) {
                usleep(500);
            }
        }
    }
    printf("SMs at jmptable\n");
#ifdef CRICKET_PROFILE
    // d-e = PROFILE tojmptbl
    gettimeofday(&d, NULL);
#endif

    // Now restore the thread-local states
    for (int sm = 0; sm != dev_prop.numSMs; sm++) {
        printf("sm %d\n", sm);
        res = cudbgAPI->readValidWarps(warp_info.dev, sm, &warp_mask);
        if (res != CUDBG_SUCCESS) {
            printf("%d:", __LINE__);
            goto cuda_error;
        }
        if (warp_mask == 0) {
            continue;
        }
        warp_info.sm = sm;

        for (uint64_t warp = 0; warp != dev_prop.numWarps; warp++) {
            if (warp_mask & (1LU << warp)) {
                printf("\twarp %lu\n", warp);

                cur_address = start_address;

                warp_info.warp = warp;

                res = cudbgAPI->readValidLanes(warp_info.dev, sm, warp,
                                               &lanemask);
                if (res != CUDBG_SUCCESS) {
                    printf("%d:", __LINE__);
                    goto cuda_error;
                }
                // Write Predicate 1 to 1 so that we enter the jumptable
                for (uint32_t lane = 0; lane != dev_prop.numLanes; lane++) {
                    if (lanemask & (1LU << lane)) {
                        res = cudbgAPI->writePredicates(0, sm, warp, lane, 1,
                                                        &predicate);
                        if (res != CUDBG_SUCCESS) {
                            fprintf(stderr, "cricket-cr (%d): %s\n", __LINE__,
                                    cudbgGetErrorString(res));
                            goto detach;
                        }
                    }
                }
                // Enter the jumptable
                res = cudbgAPI->singleStepWarp(0, sm, warp, 1, &sswarps);
                if (res != CUDBG_SUCCESS) {
                    printf("%d:", __LINE__);
                    goto cuda_error;
                }
                // Double check if we are still where we think we are
                cudbgAPI->readPC(0, sm, warp, 0, &rb_address);
                printf("\tcur_address: %lx, rb address: %lx\n", cur_address,
                       rb_address);
                index = kernelindex;

                // If this warp had already finished execution when the
                // checkpoint was created, it will not be present in the
                // checkpoint file, i.e., there is no callstack for this warp.
                // We let these warps jump to the final return statement and
                // execute it so they will finish immediately.
                if (!cricket_cr_read_pc(&warp_info, CRICKET_CR_NOLANE, ckp_dir,
                                        &callstack)) {
                    printf("cricket-cr: did not find callstack. exiting "
                           "warp...\n");
                    for (uint32_t lane = 0;
                         lane != warp_info.dev_prop->numLanes; lane++) {
                        if (lanemask & (1LU << lane)) {
                            res = cudbgAPI->writeRegister(
                                warp_info.dev, warp_info.sm, warp_info.warp,
                                lane, CRICKET_JMX_ADDR_REG,
                                ((uint32_t)(index->exit_address - cur_address -
                                            0x8)));
                            if (res != CUDBG_SUCCESS) {
                                fprintf(stderr, "cricket-cr (%d): %s\n",
                                        __LINE__, cudbgGetErrorString(res));
                                goto detach;
                            }
                        }
                    }
                    res = cudbgAPI->singleStepWarp(0, sm, warp, 1, &sswarps);
                    if (res != CUDBG_SUCCESS) {
                        printf("%d:", __LINE__);
                        goto cuda_error;
                    }
                    cudbgAPI->readPC(0, sm, warp, 0, &rb_address);
                    printf("\tsuccess (exit) (%lx = %lx) \n",
                           index->exit_address, rb_address);
                    continue;
                }

                // Now restore the callstack. We need to restore the state
                // at each subcall level.
                for (int c_level = callstack.callstack_size - 1;
                     c_level + 1 > 0; --c_level) {
                    fn = callstack.function_names[c_level];
                    printf("\t\tc_level: %d, fn: %s\n", c_level, fn);

                    if (index->ssy_num > 0) {
                        printf("\t\trestoring ssy\n");
                        if (!cricket_cr_rst_ssy(cudbgAPI, &warp_info,
                                                &callstack, c_level, index->ssy,
                                                index->ssy_num, &cur_address)) {
                            fprintf(stderr, "error restoring SSY\n");
                            goto detach;
                        }
                        cudbgAPI->readPC(0, sm, warp, 0, &rb_address);
                        printf("\tcur_address: %lx, rb address: %lx\n",
                               cur_address, rb_address);
                        printf("\t\tsuccess (ssy)\n");
                    }
                    // If there is another subcall level, we need to jump to
                    // the jumptable in the called function and enter it.
                    if (c_level > 0) {
                        printf("\t\trestoring subcall\n");
                        if (!cricket_cr_rst_subcall(
                                 cudbgAPI, &warp_info, &callstack, c_level,
                                 index->cal, index->cal_num, &cur_address)) {
                            fprintf(stderr, "error restoring CAL\n");
                            goto detach;
                        }

                        if (!cricket_elf_get_jmptable_index(
                                 jmptbl, jmptbl_len,
                                 callstack.function_names[c_level - 1],
                                 &index)) {
                            fprintf(stderr, "get jmptable entry failed\n");
                            goto detach;
                        }
                        if (index == NULL) {
                            jmptable_addr = callstack.pc[c_level - 1].absolute;
                            cur_address = callstack.pc[c_level - 1].relative;
                        } else {
                            jmptable_addr = callstack.pc[c_level - 1].absolute -
                                            callstack.pc[c_level - 1].relative +
                                            index->start_address + 0x8;
                            cur_address = index->start_address + 0x8;
                        }

                        for (uint32_t lane = 0;
                             lane != warp_info.dev_prop->numLanes; lane++) {
                            if (lanemask & (1LU << lane)) {
                                res = cudbgAPI->writeRegister(
                                    warp_info.dev, warp_info.sm, warp_info.warp,
                                    lane, CRICKET_JMX_ADDR_REG,
                                    ((uint32_t)(jmptable_addr)));
                                if (res != CUDBG_SUCCESS) {
                                    fprintf(stderr, "cricket-cr (%d): %s\n",
                                            __LINE__, cudbgGetErrorString(res));
                                    goto detach;
                                }
                            }
                        }

                        res = cudbgAPI->singleStepWarp(0, sm, warp, 1, &sswarps);
                        if (res != CUDBG_SUCCESS) {
                            printf("%d:", __LINE__);
                            goto cuda_error;
                        }
                        cudbgAPI->readPC(0, sm, warp, 0, &rb_address);
                        printf("\tcur_address: %lx, rb address: %lx\n",
                               cur_address, rb_address);
                        printf("\t\tsuccess (subcall)\n");
                        if (c_level - 1 == 0 &&
                            cur_address == callstack.pc[c_level - 1].relative) {
                            break;
                        }
                    }
                }
                // if a warp has had diverged lanes/threads, we need to diverge
                // them again using the SSY and SYNC instructions
                int predicate_value;
                if (callstack.active_lanes != callstack.valid_lanes) {
                    for (uint32_t lane = 0;
                         lane != warp_info.dev_prop->numLanes; lane++) {
                        if (lanemask & (1LU << lane)) {
                            res = cudbgAPI->writeRegister(
                                warp_info.dev, warp_info.sm, warp_info.warp,
                                lane, CRICKET_JMX_ADDR_REG,
                                ((uint32_t)(index->sync_address - cur_address -
                                            0x8)));
                            if (res != CUDBG_SUCCESS) {
                                fprintf(stderr, "cricket-cr (%d): %s\n",
                                        __LINE__, cudbgGetErrorString(res));
                                goto detach;
                            }
                            if (callstack.active_lanes & (1LU << lane)) {
                                predicate_value = 0;
                            } else {
                                predicate_value = 1;
                            }
                            res = cudbgAPI->writePredicates(
                                0, sm, warp, lane, 1, &predicate_value);
                            if (res != CUDBG_SUCCESS) {
                                fprintf(stderr, "cricket-cr (%d): %s\n",
                                        __LINE__, cudbgGetErrorString(res));
                                goto detach;
                            }
                        }
                    }
                    res = cudbgAPI->singleStepWarp(0, sm, warp, 1, &sswarps);
                    if (res != CUDBG_SUCCESS) {
                        printf("%d:", __LINE__);
                        goto cuda_error;
                    }
                    res = cudbgAPI->singleStepWarp(0, sm, warp, 1, &sswarps);
                    if (res != CUDBG_SUCCESS) {
                        printf("%d:", __LINE__);
                        goto cuda_error;
                    }
                    cur_address = index->sync_address + 0x8;
                    cudbgAPI->readPC(0, sm, warp, 0, &rb_address);
                    printf("\tsuccess (sync) (%lx = %lx) \n", cur_address,
                           rb_address);
                    cudbgAPI->readPC(0, sm, warp, 1, &rb_address);
                    printf("\tsuccess (sync2) (%lx = %lx) \n", cur_address,
                           rb_address);
                    // double check
                    uint32_t al, vl;
                    cudbgAPI->readActiveLanes(0, sm, warp, &al);
                    cudbgAPI->readValidLanes(0, sm, warp, &vl);
                    printf("valid: %x, active: %x, goal: %x\n", vl, al,
                           callstack.active_lanes);
                }
                // Restore PC
                if (cur_address != callstack.pc[0].relative) {
                    printf("\tjumping to checkpointed PC %lx\n",
                           callstack.pc[0].relative);
                    for (uint32_t lane = 0;
                         lane != warp_info.dev_prop->numLanes; lane++) {
                        if (lanemask & (1LU << lane)) {
                            res = cudbgAPI->writeRegister(
                                warp_info.dev, warp_info.sm, warp_info.warp,
                                lane, CRICKET_JMX_ADDR_REG,
                                ((uint32_t)(callstack.pc[0].relative -
                                            cur_address - 0x8)));
                            if (res != CUDBG_SUCCESS) {
                                fprintf(stderr, "cricket-cr (%d): %s\n",
                                        __LINE__, cudbgGetErrorString(res));
                                goto detach;
                            }
                        }
                    }
                    res = cudbgAPI->singleStepWarp(0, sm, warp, 1, &sswarps);
                    if (res != CUDBG_SUCCESS) {
                        printf("%d:", __LINE__);
                        goto cuda_error;
                    }
                    cudbgAPI->readPC(0, sm, warp, 0, &rb_address);
                    printf("\tsuccess (pc) (%lx = %lx) \n",
                           callstack.pc[0].relative, rb_address);

                } else {
                    // If there is no jumptable for a subcall, the threads
                    // must have been stopped at the start of the function
                    // (guaranteed by checkpoint procedure) so we do not
                    // have to do anything else here.
                    printf("\tlowest call level has no jmptable. restored to "
                           "PC %lx\n",
                           callstack.pc[0].relative);
                }
            }
        }
    }

#ifdef CRICKET_PROFILE
    // e-f = PROFILE globals
    gettimeofday(&e, NULL);
#endif

    // Now we restore kernel global data, i.e., global variables and
    // parameters.
    uint64_t pc_rb;

    cricket_elf_info elf_info;
    cricket_elf_get_info(kernel_name, &elf_info);
    printf("cricket: stack-size: %u, param-addr: %u, param-size: %u\n",
           elf_info.stack_size, elf_info.param_addr, elf_info.param_size);

    if (!cricket_cr_rst_globals(cudbgAPI, ckp_dir)) {
        printf("cricket: global variable memory restores unsuccessful\n");
    } else {
        printf("cricket: restored global variables\n");
    }

    if (!cricket_cr_rst_params(cudbgAPI, ckp_dir, &elf_info, 0, 0, 0)) {
        printf("cricket: parameter restore unsuccessful\n");
    } else {
        printf("cricket: restored parameters\n");
    }

    warp_info.dev = 0;
    warp_info.dev_prop = &dev_prop;

#ifdef CRICKET_PROFILE
    // f-g = PROFILE inkernel
    gettimeofday(&f, NULL);
#endif

    for (int sm = 0; sm != dev_prop.numSMs; sm++) {
        res = cudbgAPI->readValidWarps(warp_info.dev, sm, &warp_mask);
        if (res != CUDBG_SUCCESS) {
            printf("%d:", __LINE__);
            goto cuda_error;
        }
        for (uint64_t warp = 0; warp != dev_prop.numWarps; warp++) {
            if (warp_mask & (1LU << warp)) {
                cudbgAPI->readActiveLanes(0, sm, warp, &active_lanes);

                res = cudbgAPI->readValidLanes(warp_info.dev, sm, warp,
                                               &lanemask);
                if (res != CUDBG_SUCCESS) {
                    printf("%d:", __LINE__);
                    goto cuda_error;
                }
                warp_info.sm = sm;
                warp_info.warp = warp;
                warp_info.kernel_name = kernel_name;
                warp_info.stack_size = elf_info.stack_size;

                // Shared memory has to be restored into individual SMs
                if (!cricket_cr_rst_shared(cudbgAPI, ckp_dir, &elf_info, 0, sm,
                                           warp)) {
                    printf("cricket: shared memory restore unsuccessful (size: "
                           "%ld)\n",
                           elf_info.shared_size);
                    continue;
                } else {
                    printf("cricket: restored shared memory\n");
                }

                // Restore thread local data, i.e., registers, local memory,etc.
                for (uint32_t lane = 0; lane != dev_prop.numLanes; lane++) {
                    if (lanemask & (1LU << lane)) {
                        // cudbgAPI->readPC(0, warp,sm,lane, &pc_rb);
                        // if not active do not say readback is incorrect, if
                        // relative%4==0
                        // then rb is after the control instruction
                        /*if (!(active_lanes & (1<<lane))) {
                            printf("cricket: lane %u not active (PC: %lx)\n",
                        lane, pc_rb);
                        } else if ((callstack.pc[0].relative % (4*8) == 0 &&
                        pc_rb !=
                        callstack.pc[0].relative+0x8) ||
                                   (callstack.pc[0].relative % (4*8) != 0 &&
                        pc_rb !=
                        callstack.pc[0].relative)) {
                            printf("cricket: lane %u: readback PC (%lx) is not
                        correct
                        (%lx)\n", lane, pc_rb, callstack.pc[0].relative);
                        }*/

                        cricket_cr_rst_lane(cudbgAPI, &warp_info, lane,
                                            ckp_dir);
                    }
                }
                // printf("cricket: verified restored PCs\n");
                printf("cricket: restored warp D%uS%uW%lu\n", 0, sm, warp);
            }
        }
    }
#ifdef CRICKET_PROFILE
    gettimeofday(&g, NULL);
    double bt = ((double)((b.tv_sec * 1000000 + b.tv_usec) -
                          (a.tv_sec * 1000000 + a.tv_usec))) /
                1000000.;
    double ct = ((double)((c.tv_sec * 1000000 + c.tv_usec) -
                          (b.tv_sec * 1000000 + b.tv_usec))) /
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
    double gt = ((double)((g.tv_sec * 1000000 + g.tv_usec) -
                          (f.tv_sec * 1000000 + f.tv_usec))) /
                1000000.;
    double comt = ((double)((g.tv_sec * 1000000 + g.tv_usec) -
                            (a.tv_sec * 1000000 + a.tv_usec))) /
                  1000000.;
    printf("complete time:\n\tPROFILE patch: %f s\n\tPROFILE runattach: %f "
           "s\n\tPROFILE tojmptbl: %f s\n\tPROFILE jmptbl: %f s\n\tPROFILE "
           "globals: %f s\n\tPROFILE inkernel %f s\n\tPROFILE complete: %f s\n",
           bt, ct, dt, et, ft, gt, comt);
#endif
    // The comment below is for single stepping threads a bit further to check
    // if everything was restored correctly. If something was restored wrong,
    // it often only shows after a few instructions. Error reporting without
    // single stepping is not exact enough to determine what went wrong.
    /*
        CUDBGException_t exc = 0;
        uint64_t errorPC;
        uint64_t virtpc;
        bool valid;
        uint32_t val;

        for (int i=0; i != 60; ++i) {
            res = cudbgAPI->singleStepWarp(0,0,first_warp,&sswarps);
            if (res != CUDBG_SUCCESS) {
                printf("%d:", __LINE__);
                goto cuda_error;
            }

            res = cudbgAPI->readPC(0,0,first_warp,0,&pc_rb);
            res = cudbgAPI->readVirtualPC(0,0,first_warp,0,&virtpc);
            printf("pc: %lx, %lx\n", pc_rb, virtpc);

            for (uint32_t lane=0; lane != 16; lane++) {
                    cudbgAPI->readLaneException(0, 0, first_warp, lane, &exc);
                    cudbgAPI->readErrorPC(0,0,first_warp,&errorPC, &valid);
                    if (exc != 0) printf("%lx: lane %u: %u, errorPC: %llx,
       %u\n",
       pc_rb, lane, exc, errorPC, valid);
            }
        }*/
    printf("resuming device...\n");
detach:
    cricket_elf_free_jumptable(&jmptbl, jmptbl_len);
    cricket_cr_free_callstack(&callstack);
    /* Detach from process (CPU and GPU) */
    detach_command(NULL, !batch_flag);
    /* finalize, i.e. clean up CUDA debugger API */
    cuda_api_finalize();
    /* quit GDB. TODO: Why is this necccessary? */
    quit_force(NULL, 0);
    return 0;
cuda_error:
    printf("Cuda Error: \"%s\"\n", cudbgGetErrorString(res));
    return -1;
}

int cricket_checkpoint(int argc, char *argv[])
{
    char *ckp_dir = "/tmp/cricket-ckp";
    uint32_t numDev = 0;
    const char *kernel_name = NULL;
    const char *warp_kn;
    uint64_t *warp_mask;
    uint32_t calldepth;
    uint64_t relative_ssy;
    cricket_callstack callstack;
    cricket_function_info *function_info = NULL;
    uint32_t first_warp = 0;
    size_t fi_num;
    int ret = -1;
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

int cricket_start(int argc, char *argv[])
{
    struct cmd_list_element *alias = NULL;
    struct cmd_list_element *prefix_cmd = NULL;
    struct cmd_list_element *cmd = NULL;
    CUDBGResult res;
    if (argc != 3) {
        printf("wrong number of arguments, use: %s <executable>\n", argv[0]);
        return -1;
    }

    cricket_init_gdb(argv[0]);

    /* load files */
    exec_file_attach(argv[2], !batch_flag);
    symbol_file_add_main_adapter(argv[2], !batch_flag);

    struct cmd_list_element *c;
    //char *pset = "set cuda break_on_launch all";
    //c = lookup_cmd(&pset, cmdlist, "", 0, 1);
    //do_set_command(pset, !batch_flag, c);

    if (!lookup_cmd_composition("run", &alias, &prefix_cmd, &cmd)) {
        printf("error looking up command run");
    }
    cmd_func(cmd, "", 0);

detach:
    /* Detach from process (CPU and GPU) */
    detach_command(NULL, !batch_flag);
    /* quit GDB. TODO: Why is this necccessary? */
    quit_force(NULL, 0);
    return 0;
}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        fprintf(stderr, "wrong number of arguments, use: %s "
                        "(start|checkpoint|restore)\n",
                argv[0]);
        return -1;
    }
    if (strcmp(argv[1], "start") == 0)
        return cricket_start(argc, argv);
    if (strcmp(argv[1], "checkpoint") == 0)
        return cricket_checkpoint(argc, argv);
    if (strcmp(argv[1], "restore") == 0 || strcmp(argv[1], "restart") == 0)
        return cricket_restore(argc, argv);
    if (strcmp(argv[1], "analyze") == 0)
        return cricket_analyze(argc, argv);
    if (strcmp(argv[1], "info") == 0)
        return cricket_info(argc, argv);

    fprintf(stderr, "Unknown operation \"%s\".\n", argv[1]);
    return -1;
}

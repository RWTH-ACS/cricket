#include "cricket-utils.h"
#include "defs.h"
#include <stdio.h>
#include <string.h>
#include <cudadebugger.h>
#include <math.h>
#include "interps.h"
#include "top.h"
#include "main.h"

void cricket_error_unreachable(void)
{
    printf("ERROR 2200: We've reached an unreachable state. Anything is possible. The limits were in our heads all along. Follow your dreams.\n");
}

double time_diff_sec(const struct timeval *tv1, const struct timeval *tv2)
{
    return fabs((tv2->tv_sec - tv1->tv_sec) +
                ((tv2->tv_usec - tv1->tv_usec) / 1000000.0));
}

uint time_diff_usec(const struct timeval *tv1, const struct timeval *tv2)
{
    return abs((tv2->tv_sec - tv1->tv_sec) * 1000000 + tv2->tv_usec -
               tv1->tv_usec);
}

void print_binary32(uint32_t num)
{
    uint8_t i;
    for (i = 0; i != 32; ++i) {
        if (num & (1LLU << 31))
            printf("1");
        else
            printf("0");
        num <<= 1;
    }
    printf("\n");
}

void print_binary64(uint64_t num)
{
    uint8_t i;
    for (i = 0; i != 64; ++i) {
        if (num & (1LLU << 63))
            printf("1");
        else
            printf("0");
        num <<= 1;
    }
    printf("\n");
}

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

// in defs.h these variables are refferd to as external, so let's provide these
// as global state. (What should possibly go wrong)
struct ui_file *gdb_stdout;
struct ui_file *gdb_stderr;
struct ui_file *gdb_stdlog;
struct ui_file *gdb_stdin;
struct ui_file *gdb_stdtarg;
struct ui_file *gdb_stdtargerr;
struct ui_file *gdb_stdtargin;

bool cricket_init_gdb(char *name)
{
    // TODO check if tools succeed
    /* initialize gdb streams, necessary for gdb_init */
    gdb_stdout = ui_file_new();
    gdb_stderr = stdio_fileopen(stderr);
    gdb_stdlog = gdb_stderr;
    gdb_stdtarg = gdb_stderr;
    gdb_stdin = stdio_fileopen(stdin);
    gdb_stdtargerr = gdb_stderr;
    gdb_stdtargin = gdb_stdin;
    instream = fopen("/dev/null", "r");

    if (! getcwd (gdb_dirbuf, sizeof (gdb_dirbuf))) {
        /* Don't use *_filtered or warning() (which relies on
           current_target) until after initialize_all_files().  */
        fprintf(stderr, "%s: warning: error finding working directory: %s\n",
                       name, safe_strerror (errno));
    }
    
    current_directory = gdb_dirbuf;

    /* initialize gdb paths */
    gdb_sysroot = strdup("");
    debug_file_directory = strdup(DEBUGDIR);
    gdb_datadir = strdup(GDB_DATADIR);

    /* tell gdb that we do not want to run an interactive shell */
    batch_flag = 1;

    /* initialize BFD, the binary file descriptor library */
    bfd_init();

    /* initialize GDB */
    printf("gdb_init...\n");
    gdb_init(name);

    char *interpreter_p = strdup(INTERP_CONSOLE);
    struct interp *interp = interp_lookup(interpreter_p);
    interp_set(interp, 1);
    return true;
}

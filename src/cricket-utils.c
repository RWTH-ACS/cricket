#include "cricket-utils.h"
#include <stdio.h>
#include <cudadebugger.h>

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

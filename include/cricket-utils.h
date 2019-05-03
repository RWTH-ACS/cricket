#ifndef _CRICKET_UTILS_H
#define _CRICKET_UTILS_H

/*
 * Helper Functions for cricket
 */

#include <stdio.h>
#include <cudadebugger.h>
#include "cricket-types.h"

#define print_error(fmt, ...)                                                  \
    do {                                                                       \
        fprintf(stderr, "Error in %s:%d:%s(): " fmt, __FILE__, __LINE__,       \
                __func__, ##__VA_ARGS__);                                      \
    } while (0)

void print_binary32(uint32_t num);

void print_binary64(uint64_t num);

bool cricket_print_lane_states(CUDBGAPI cudbgAPI, CricketDeviceProp *dev_prop);

bool cricket_all_warps_broken(CUDBGAPI cudbgAPI, CricketDeviceProp *dev_prop);

#endif // _CRICKET_UTILS_H

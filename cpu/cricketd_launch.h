#ifndef _CRICKET_LAUNCH_H_
#define _CRICKET_LAUNCH_H_

#include <stdint.h>


int cricketd_launch_load_elf(const char *filename, void **fatbin, size_t *fatbin_size);

struct __fatCubin {
    uint32_t magic;
    uint32_t seq;
    uint64_t text;
    uint64_t data;
    uint64_t ptr;
    uint64_t ptr2;
    uint64_t zero;
};

#endif //_CRICKET_LAUNCH_H_

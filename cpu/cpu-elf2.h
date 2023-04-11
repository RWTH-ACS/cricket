#ifndef _ELF_H_
#define _ELF_H_

#include <stdint.h>
#include "cpu-common.h"
#include "list.h"

struct fat_header {
    uint32_t magic;
    uint32_t version;
    uint64_t text;
    uint64_t data;  // points to outside of the file
    uint64_t unknown;
    uint64_t text2;     // points to footer of text section
    uint64_t zero;
};

int elf2_init(void);
int elf2_get_fatbin_info(struct fat_header *fatbin, list *kernel_infos, void** fatbin_mem, unsigned* fatbin_size);

int elf2_parameter_info(list *kernel_infos, void* memory, size_t memsize);
void* elf2_symbol_address(const char* file, char *symbol);
int elf2_contains_kernel(void* memory, size_t memsize);

#endif //_ELF_H_

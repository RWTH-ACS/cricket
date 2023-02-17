#ifndef _CPU_UTILS_H_
#define _CPU_UTILS_H_

#include <stdint.h>
#include "cpu-common.h"
#include "list.h"

struct fat_header {
    uint32_t magic;
    uint32_t version;
    uint64_t text;
    uint64_t data;  // points to outside of the file
    uint64_t unknown;
    uint64_t text2;
    uint64_t zero;
};


int cpu_utils_get_fatbin_info(struct fat_header *fatbin, void** fatbin_mem, unsigned* fatbin_size);

void kernel_infos_free(kernel_info_t *infos, size_t kernelnum);


int cpu_utils_is_local_connection(struct svc_req *rqstp);
int cpu_utils_command(char **command);
int cpu_utils_md5hash(char *filename, unsigned long *high, unsigned long *low);
void* cricketd_utils_symbol_address(const char* file, char *symbol);
int cricketd_utils_launch_child(const char *file, char **args);
int cpu_utils_parameter_info(list *kernel_infos, char *path);
int cpu_utils_contains_kernel(const char *path);
kernel_info_t* cricketd_utils_search_info(list *kernel_infos, char *kernelname);

#endif //_CPU_UTILS_H_

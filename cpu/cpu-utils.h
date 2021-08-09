#ifndef _CPU_UTILS_H_
#define _CPU_UTILS_H_

#include <stdint.h>
#include "cpu-common.h"
#include "list.h"

void kernel_infos_free(kernel_info_t *infos, size_t kernelnum);


int cpu_utils_is_local_connection(struct svc_req *rqstp);
int cpu_utils_command(char **command);
int cpu_utils_md5hash(char *filename, unsigned long *high, unsigned long *low);
void* cricketd_utils_symbol_address(char *symbol);
int cricketd_utils_launch_child(const char *file, char **args);
int cpu_utils_parameter_info(list *kernel_infos, char *path);
int cpu_utils_contains_kernel(const char *path);
kernel_info_t* cricketd_utils_search_info(list *kernel_infos, char *kernelname);

#endif //_CPU_UTILS_H_

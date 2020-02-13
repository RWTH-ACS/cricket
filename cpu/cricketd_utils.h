#ifndef _CRICKETD_UTILS_H_
#define _CRICKETD_UTILS_H_

typedef struct kernel_info {
    char *name;
    size_t param_size;
} kernel_info_t;

void kernel_infos_free(kernel_info_t *infos, size_t kernelnum);


void* cricketd_utils_symbol_address(char *symbol);
int cricketd_utils_launch_child(const char *file, char **args);
int cricketd_utils_parameter_size(kernel_info_t **infos, size_t *kernelnum);
kernel_info_t* cricketd_utils_search_info(kernel_info_t *infos, size_t kernelnum, char *kernelname);

#endif //_CRICKETD_UTILS_H

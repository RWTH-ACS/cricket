#ifndef _CPU_SERVER_NVML_H_
#define _CPU_SERVER_NVML_H_

int server_nvml_init(int restore);
int server_nvml_deinit(void);
//int server_nvml_checkpoint(const char *path, int dump_memory, unsigned long prog, unsigned long vers);
//int server_nvml_restore(const char *path);

#endif //_CPU_SERVER_NVML_H_

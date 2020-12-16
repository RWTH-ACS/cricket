#ifndef _CPU_SERVER_RUNTIME_H_
#define _CPU_SERVER_RUNTIME_H_

int server_runtime_init(int restore);
int server_runtime_deinit(void);
int server_runtime_checkpoint(const char *path, int dump_memory, unsigned long prog, unsigned long vers);
int server_runtime_restore(const char *path);

#endif //_CPU_SERVER_RUNTIME_H_

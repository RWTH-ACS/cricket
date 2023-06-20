#ifndef _CPU_SERVER_CUDNN_H_
#define _CPU_SERVER_CUDNN_H_

#include "resource-mg.h"

int server_cudnn_init(int restore);
int server_cudnn_deinit(void);

#endif // _CPU_SERVER_CUDNN_H_
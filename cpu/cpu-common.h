#ifndef _CD_COMMON_H_
#define _CD_COMMON_H_

#include <rpc/rpc.h>

#define CD_SOCKET_PATH "/tmp/cricketd_sock"
#define CRICKET_PATH "/home/eiling/projects/cricket/gpu/cricket"

typedef struct kernel_info {
    char *name;
    size_t param_size;
    size_t param_num;
    uint16_t *param_offsets;
    uint16_t *param_sizes;
    void *host_fun;
} kernel_info_t;

enum socktype_t {UNIX, TCP, UDP} socktype;
#define INIT_SOCKTYPE enum socktype_t socktype = TCP;


CLIENT *clnt;

size_t kernelnum;
kernel_info_t *infos;

#endif //_CD_COMMON_H_


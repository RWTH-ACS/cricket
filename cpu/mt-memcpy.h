#ifndef _MT_MEMCPY_H_
#define _MT_MEMCPY_H_

#include <pthread.h>
#include <stdint.h>
#include "oob.h"

typedef struct _mt_memcpy_server_t {
    pthread_t server_thread;
    void* dev_ptr;
    void* mem_ptr;
    size_t mem_size;
    int thread_num;
    oob_t oob;
    uint16_t port;
    pthread_barrier_t barrier;
} mt_memcpy_server_t;

int mt_memcpy_init_server(mt_memcpy_server_t *server, void* dev_ptr, size_t size, uint16_t port);

int mt_memcpy_sync_server(mt_memcpy_server_t *server);


#endif // _MT_MEMCPY_H_

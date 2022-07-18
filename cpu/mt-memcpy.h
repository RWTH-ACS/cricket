#ifndef _MT_MEMCPY_H_
#define _MT_MEMCPY_H_

#include <pthread.h>
#include <stdint.h>
#include "oob.h"

enum mt_memcpy_direction {
    MT_MEMCPY_HTOD,
    MT_MEMCPY_DTOH
};

typedef struct _mt_memcpy_server_t {
    pthread_t server_thread;
    void* dev_ptr;
    void* mem_ptr;
    size_t mem_size;
    int thread_num;
    oob_t oob;
    uint16_t port;
    pthread_barrier_t barrier;
    enum mt_memcpy_direction dir;
} mt_memcpy_server_t;

int mt_memcpy_init_server(mt_memcpy_server_t *server, void* dev_ptr, size_t size, enum mt_memcpy_direction dir, int thread_num);

int mt_memcpy_sync_server(mt_memcpy_server_t *server);
int mt_memcpy_client(const char* server, uint16_t port, void* host_ptr, size_t size, enum mt_memcpy_direction dir, int thread_num);


#endif // _MT_MEMCPY_H_

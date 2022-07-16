#include <stdint.h>
#include <pthread.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <errno.h>
#include <string.h>

#include "mt-memcpy.h"
#include "log.h"
#include "resource-mg.h"


static void* mt_memcpy_listener_thread(void* targs)
{
    mt_memcpy_server_t *args = (mt_memcpy_server_t*)targs;
    size_t ret = 1;
    if (oob_init_listener_socket(&args->oob, args->port) != 0) {
        LOGE(LOG_ERROR, "oob: failed to initialize listener socket.");
        return (void*)ret;
    }
    int iret = pthread_barrier_wait(&args->barrier);
    if (iret != 0 && iret != PTHREAD_BARRIER_SERIAL_THREAD) {
        LOGE(LOG_ERROR, "oob: failed to wait for barrier.");
        return (void*)ret;
    }
    if (oob_init_listener_accept(&args->oob) != 0) {
        LOGE(LOG_ERROR, "oob: failed to accept connection.");
        return (void*)ret;
    }
    if ((iret = oob_receive(&args->oob, args->mem_ptr, args->mem_size)) != args->mem_size) {
        LOGE(LOG_ERROR, "oob: failed to receive memory: received %d bytes: %s", iret, strerror(errno));
        return (void*)ret;
    }

    cudaError_t res = cudaMemcpy(
      resource_mg_get(&rm_memory, args->dev_ptr),
      args->mem_ptr,
      args->mem_size,
      cudaMemcpyHostToDevice);

    cudaFreeHost(args->mem_ptr);
    if (oob_close(&args->oob) != 0) {
        LOGE(LOG_ERROR, "oob: failed to close connection.");
        return (void*)ret;
    }
    ret = 0;
    return (void*)ret;
}

int mt_memcpy_init_server(mt_memcpy_server_t *server, void* dev_ptr, size_t size, uint16_t port)
{
    cudaError_t cudaRes;
    if (server == NULL) {
        return 1;
    }
    server->dev_ptr = dev_ptr;
    server->port = port;
    server->mem_size = size;
    if (pthread_barrier_init(&server->barrier, NULL, 2) != 0) {
        LOGE(LOG_ERROR, "oob: failed to initialize barrier.");
        return 1;
    }

    if (pthread_create(&server->server_thread, NULL, mt_memcpy_listener_thread, server) != 0) {
        LOGE(LOG_ERROR, "oob: failed to create listener thread.");
        return 1;
    }

    if ((cudaRes = cudaHostAlloc(&server->mem_ptr, server->mem_size, 0)) != cudaSuccess) {
        LOGE(LOG_ERROR, "cudaHostAlloc failed: %d.", cudaRes);
        return 1;
    }
    
    int bret = pthread_barrier_wait(&server->barrier);
    if (bret != 0 && bret != PTHREAD_BARRIER_SERIAL_THREAD) {
        LOGE(LOG_ERROR, "oob: failed to wait for barrier.");
        return 1;
    }
    pthread_barrier_destroy(&server->barrier);
    return 0;
}

int mt_memcpy_sync_server(mt_memcpy_server_t *server)
{
    size_t ret;
    if (pthread_join(server->server_thread, (void**)&ret) != 0) {
        LOGE(LOG_ERROR, "oob: failed to join listener thread.");
        return 1;
    }
    return ret;
}

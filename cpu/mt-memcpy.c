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
    args->port = args->oob.port;
    LOGE(LOG_DEBUG, "oob: listening on port %d.", args->port);
    int iret = pthread_barrier_wait(&args->barrier);
    if (iret != 0 && iret != PTHREAD_BARRIER_SERIAL_THREAD) {
        LOGE(LOG_ERROR, "oob: failed to wait for barrier.");
        return (void*)ret;
    }
    if (oob_init_listener_accept(&args->oob) != 0) {
        LOGE(LOG_ERROR, "oob: failed to accept connection.");
        return (void*)ret;
    }
    if (args->dir == MT_MEMCPY_HTOD) {
        if ((iret = oob_receive(&args->oob, args->mem_ptr, args->mem_size)) != args->mem_size) {
            LOGE(LOG_ERROR, "oob: failed to receive memory: received %d bytes: %s", iret, strerror(errno));
            return (void*)ret;
        }

        cudaError_t res = cudaMemcpy(
          resource_mg_get(&rm_memory, args->dev_ptr),
          args->mem_ptr,
          args->mem_size,
          cudaMemcpyHostToDevice);
        if (res != cudaSuccess) {
            LOGE(LOG_ERROR, "oob: failed to copy memory: %s", cudaGetErrorString(res));
        }
    } else if (args->dir == MT_MEMCPY_DTOH) {
        if ((iret = oob_send(&args->oob, args->mem_ptr, args->mem_size)) != args->mem_size) {
            LOGE(LOG_ERROR, "oob: failed to send memory: sent %d bytes: %s", iret, strerror(errno));
            return (void*)ret;
        }
    }

    cudaFreeHost(args->mem_ptr);
    if (oob_close(&args->oob) != 0) {
        LOGE(LOG_ERROR, "oob: failed to close connection.");
        return (void*)ret;
    }
    ret = 0;
    return (void*)ret;
}

int mt_memcpy_init_server(mt_memcpy_server_t *server, void* dev_ptr, size_t size, enum mt_memcpy_direction dir)
{
    cudaError_t cudaRes;
    int ret = 1;
    if (server == NULL) {
        return ret;
    }
    server->dev_ptr = dev_ptr;
    server->port = 0;
    server->mem_size = size;
    server->dir = dir;
    if (pthread_barrier_init(&server->barrier, NULL, 2) != 0) {
        LOGE(LOG_ERROR, "oob: failed to initialize barrier.");
        return ret;
    }

    if (pthread_create(&server->server_thread, NULL, mt_memcpy_listener_thread, server) != 0) {
        LOGE(LOG_ERROR, "oob: failed to create listener thread.");
        goto cleanup;
    }

    if ((cudaRes = cudaHostAlloc(&server->mem_ptr, server->mem_size, 0)) != cudaSuccess) {
        LOGE(LOG_ERROR, "cudaHostAlloc failed: %d.", cudaRes);
        pthread_cancel(server->server_thread);
        goto cleanup;
    }

    if (dir == MT_MEMCPY_DTOH) {
        cudaError_t res = cudaMemcpy(
            server->mem_ptr,
            resource_mg_get(&rm_memory, dev_ptr),
            size,
            cudaMemcpyDeviceToHost);
        if (res != cudaSuccess) {
            LOGE(LOG_ERROR, "oob: failed to copy memory: %s", cudaGetErrorString(res));
            pthread_cancel(server->server_thread);
            goto cleanup;
        }
    }
    
    int bret = pthread_barrier_wait(&server->barrier);
    if (bret != 0 && bret != PTHREAD_BARRIER_SERIAL_THREAD) {
        LOGE(LOG_ERROR, "oob: failed to wait for barrier.");
        pthread_cancel(server->server_thread);
        goto cleanup;
    }
    ret = 0;
 cleanup:
    pthread_barrier_destroy(&server->barrier);
    return ret;
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

#include "sched.h"
#include "log.h"
#include "list.h"
#include <pthread.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

typedef struct _sched_none_t {
    int id;
    int device;
} sched_none_t;

static list ids;
static pthread_mutex_t mutex_device;

int sched_none_init(void)
{
    LOG(LOG_DEBUG, "sched_none_init");
    list_init(&ids, sizeof(sched_none_t));
    pthread_mutex_init(&mutex_device, NULL);
    return 0;
}

int sched_none_retain(int id)
{
    sched_none_t *elem = NULL;
    LOG(LOG_DEBUG, "sched_none_retain(%d)", id);
    for (size_t i = 0; i < ids.length; ++i) {
        elem = (sched_none_t*)list_get(&ids, i);
        if (id == elem->id) {
           break; 
        }
    }
    if (elem == NULL) {
        if (list_append(&ids, (void**)&elem) != 0) {
            LOGE(LOG_ERROR, "error adding element %d to ids list", id);
            return 1;
        }
        elem->id = id;
        elem->device = 0;
        LOGE(LOG_DEBUG, "added %d to ids list", id);
    }

    if (pthread_mutex_lock(&mutex_device) != 0) {
        LOGE(LOG_ERROR, "mutex lock failed");
        return 1;
    }

    cudaError_t err;
    if ((err = cudaSetDevice(elem->device)) != CUDA_SUCCESS) {
        LOGE(LOG_ERROR, "cudaSetDevice: %s", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int sched_none_release(int id)
{
    LOG(LOG_DEBUG, "sched_none_release(%d)", id);
    if (pthread_mutex_unlock(&mutex_device) != 0) {
        LOGE(LOG_ERROR, "mutex lock failed");
        return 1;
    }
    return 1;
}

int sched_none_rm(int id)
{
    size_t i;
    LOG(LOG_DEBUG, "sched_none_rm(%d)", id);
    for (i = 0; i < ids.length; ++i) {
        if (id == ((sched_none_t*)list_get(&ids, i))->id) {
           break; 
        }
    }
    if (i == ids.length) {
        LOGE(LOG_ERROR, "the id %d does not exist in ids list", id);
        return 1;
    } else {
        return list_rm(&ids, i);
    }
}

void sched_none_deinit(void)
{
    LOG(LOG_DEBUG, "sched_none_deinit");
    list_free(&ids);
    pthread_mutex_destroy(&mutex_device);
}

sched_t sched_none = {
    .init = sched_none_init,
    .retain = sched_none_retain,
    .release = sched_none_release,
    .rm = sched_none_rm,
    .deinit = sched_none_deinit
};

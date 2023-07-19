#include "gsched.h"
#include "log.h"
#include "list.h"
#include <pthread.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

typedef struct _gsched_none_t {
    int id;
    int device;
} gsched_none_t;

static list ids;
static pthread_mutex_t mutex_device;
static pthread_mutex_t mutex_ids;
static int cuda_max_devices;

int gsched_none_init(void)
{
    cudaError_t res;
    LOG(LOG_DEBUG, "sched_none_init");
    list_init(&ids, sizeof(gsched_none_t));
    pthread_mutex_init(&mutex_device, NULL);
    pthread_mutex_init(&mutex_ids, NULL);
    if ((res = cudaGetDeviceCount(&cuda_max_devices)) != cudaSuccess) {
        LOGE(LOG_ERROR, "cudaGetDeviceCount failed: %s (%d)", cudaGetErrorString(res), res);
        return 1;
    }
    return 0;
}

static int gsched_none_device_sched(void)
{
    static int next_device_id = 0;
    int ret = next_device_id;
    next_device_id = (next_device_id + 1) % cuda_max_devices;
    return ret;
}

int gsched_none_retain(int id)
{
    gsched_none_t *elem = NULL;
    int ret = 1;
    LOG(LOG_DEBUG, "sched_none_retain(%d)", id);

    //TODO: if ids.length == 1 bypass this.

    if (pthread_mutex_lock(&mutex_ids) != 0) {
        LOGE(LOG_ERROR, "mutex lock failed");
        return 1;
    }

    for (size_t i = 0; i < ids.length; ++i) {
        elem = (gsched_none_t*)list_get(&ids, i);
        if (id == elem->id) {
           break; 
        }
    }
    if (elem == NULL) {
        if (list_append(&ids, (void**)&elem) != 0) {
            LOGE(LOG_ERROR, "error adding element %d to ids list", id);
            ret = 1; 
            goto cleanup1;
        }
        elem->id = id;
        elem->device = gsched_none_device_sched();
        LOGE(LOG_DEBUG, "added %d to ids list", id);
    }
 cleanup1:
    if (pthread_mutex_unlock(&mutex_ids) != 0) {
        LOGE(LOG_ERROR, "mutex unlock failed");
        return 1;
    }

    if (ret != 0) {
        return ret;
    }

    if (pthread_mutex_lock(&mutex_device) != 0) {
        LOGE(LOG_ERROR, "mutex lock failed");
        return 1;
    }

    cudaError_t err;
    if ((err = cudaSetDevice(elem->device)) != CUDA_SUCCESS) {
        LOGE(LOG_ERROR, "cudaSetDevice: %s", cudaGetErrorString(err));
        if (pthread_mutex_unlock(&mutex_device) != 0) {
            LOGE(LOG_ERROR, "mutex unlock failed");
        }
        return 1;
    }
    return 0;
}

int gsched_none_release(int id)
{
    LOG(LOG_DEBUG, "sched_none_release(%d)", id);
    if (pthread_mutex_unlock(&mutex_device) != 0) {
        LOGE(LOG_ERROR, "mutex unlock failed");
        return 1;
    }
    return 1;
}

int gsched_none_rm(int id)
{
    size_t i;
    int ret = 1;
    LOG(LOG_DEBUG, "sched_none_rm(%d)", id);

    if (pthread_mutex_lock(&mutex_ids) != 0) {
        LOGE(LOG_ERROR, "mutex lock failed");
        return 1;
    }

    for (i = 0; i < ids.length; ++i) {
        if (id == ((gsched_none_t*)list_get(&ids, i))->id) {
           break; 
        }
    }
    if (i == ids.length) {
        LOGE(LOG_ERROR, "the id %d does not exist in ids list", id);
        ret = 1;
        goto cleanup;
    } else {
        ret = list_rm(&ids, i);
        goto cleanup;
    }
 cleanup:
    if (pthread_mutex_unlock(&mutex_ids) != 0) {
        LOGE(LOG_ERROR, "mutex unlock failed");
        return 1;
    }
    return ret;

}

void gsched_none_deinit(void)
{
    LOG(LOG_DEBUG, "sched_none_deinit");
    list_free(&ids);
    pthread_mutex_destroy(&mutex_device);
    pthread_mutex_destroy(&mutex_ids);
}

gsched_t sched_none = {
    .init = gsched_none_init,
    .retain = gsched_none_retain,
    .release = gsched_none_release,
    .rm = gsched_none_rm,
    .deinit = gsched_none_deinit
};

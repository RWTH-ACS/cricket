#define _GNU_SOURCE
#include <cuda_runtime_api.h>
#include <cuda.h>

//for strerror
#include <string.h>
#include <errno.h>

//For SHM
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "cpu_rpc_prot.h"
#include "cpu-common.h"
#include "cpu-utils.h"
#include "log.h"
#include "list.h"
#include "rpc/types.h"
#ifdef WITH_IB
#include <pthread.h>
#include "cpu-ib.h"
#endif //WITH_IB

#define WITH_RECORDER
#include "api-recorder.h"
#include "resource-mg.h"
#include "cpu-server-runtime.h"
#include "cr.h"
#include "cpu-server-cusolver.h"
#include "cpu-server-cublas.h"

typedef struct host_alloc_info {
    int cnt;
    size_t size;
    void *client_ptr;
    void *server_ptr;
} host_alloc_info_t;
static host_alloc_info_t hainfo[64];
static size_t hainfo_cnt = 1;

static resource_mg rm_streams;
static resource_mg rm_events;
static resource_mg rm_arrays;
static resource_mg rm_memory;

static int hainfo_getserverindex(void *server_ptr)
{
    int i;
    for (i=0; i < hainfo_cnt; ++i) {
        if (hainfo[i].server_ptr != 0 &&
            hainfo[i].server_ptr == server_ptr) {

            return i;
        }
    }
    return -1;
}

int server_runtime_init(int restore)
{
    #ifdef WITH_IB
    #endif //WITH_IB
   
    int ret = 0;
    ret = list_init(&api_records, sizeof(api_record_t));
    if (!restore) {
        ret &= resource_mg_init(&rm_streams, 1);
        ret &= resource_mg_init(&rm_events, 1);
        ret &= resource_mg_init(&rm_arrays, 1);
        ret &= resource_mg_init(&rm_memory, 1);
        ret &= cusolver_init(1, &rm_streams, &rm_memory);
        ret &= cublas_init(1, &rm_memory);
    } else {
        ret &= resource_mg_init(&rm_streams, 0);
        ret &= resource_mg_init(&rm_events, 0);
        ret &= resource_mg_init(&rm_arrays, 0);
        ret &= resource_mg_init(&rm_memory, 0);
        ret &= cusolver_init(0, &rm_streams, &rm_memory);
        ret &= cublas_init(0, &rm_memory);
        ret &= server_runtime_restore("ckp");
    }
    return ret;
}

int server_runtime_deinit(void)
{
    //api_records_print();
    api_records_free_args();
    list_free(&api_records);
    resource_mg_free(&rm_streams);
    resource_mg_free(&rm_events);
    resource_mg_free(&rm_arrays);
    resource_mg_free(&rm_memory);
    cusolver_deinit();
    cublas_deinit();
    return 0;

}

int server_runtime_checkpoint(const char *path, int dump_memory, unsigned long prog, unsigned long vers)
{
    if (cr_dump_rpc_id(path, prog, vers) != 0) {
        LOGE(LOG_ERROR, "error dumping api_records");
        return 1;
    }
    if (cr_dump(path) != 0) {
        LOGE(LOG_ERROR, "error dumping api_records");
        return 1;
    }
    if (dump_memory == 1) {
        if (cr_dump_memory(path) != 0) {
            LOGE(LOG_ERROR, "error dumping memory");
            return 1;
        }
    }
    return 0;
}

int server_runtime_restore(const char *path)
{
    struct timeval start, end;
    double time = 0;
    gettimeofday(&start, NULL);
    if (cr_restore(path, &rm_memory, &rm_streams, &rm_events, &rm_arrays, cusolver_get_rm(), cublas_get_rm()) != 0) {
        LOGE(LOG_ERROR, "error restoring api_records");
        return 1;
    }
    gettimeofday(&end, NULL);
    time = ((double)((end.tv_sec * 1e6 + end.tv_usec) -
                     (start.tv_sec * 1e6 + start.tv_usec)))/1.e6;
    LOGE(LOG_INFO, "time: %f", time);

    return 0;
}

/* ############### RUNTIME API ############### */
/* ### Device Management ### */
bool_t cuda_choose_device_1_svc(mem_data prop, int_result *result, struct svc_req *rqstp)
{
    struct cudaDeviceProp *cudaProp;
    LOGE(LOG_DEBUG, "cudaChooseDevice");
    if (prop.mem_data_len != sizeof(struct cudaDeviceProp)) {
        LOGE(LOG_ERROR, "Received wrong amount of data: expected %zu but got %zu", sizeof(struct cudaDeviceProp), prop.mem_data_len);
        return 0;
    }
    cudaProp = (struct cudaDeviceProp*)prop.mem_data_val;
    RECORD_API(struct cudaDeviceProp);
    RECORD_SINGLE_ARG(*cudaProp);

    result->err = cudaChooseDevice(&result->int_result_u.data, cudaProp);

    RECORD_RESULT(integer, result->int_result_u.data);
    return 1;
}

bool_t cuda_device_get_attribute_1_svc(int attr, int device, int_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cudaDeviceGetAttribute");
    result->err = cudaDeviceGetAttribute(&result->int_result_u.data, (enum cudaDeviceAttr)attr, device);
    return 1;
}

bool_t cuda_device_get_by_pci_bus_id_1_svc(char* pciBusId, int_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cudaDeviceGetByPCIBusId");
    result->err = cudaDeviceGetByPCIBusId(&result->int_result_u.data, pciBusId);
    return 1;
}

bool_t cuda_device_get_cache_config_1_svc(int_result *result, struct svc_req *rqstp)
{
    enum cudaFuncCache res;
    LOGE(LOG_DEBUG, "cudaDeviceGetCacheConfig");
    result->err = cudaDeviceGetCacheConfig(&res);
    result->int_result_u.data = (int)res;
    return 1;
}

bool_t cuda_device_get_limit_1_svc(int limit, u64_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cudaDeviceLimit");
    result->err = cudaDeviceGetLimit(&result->u64_result_u.u64, limit);
    return 1;
}

//        /*mem_result CUDA_DEVICE_GET_NVSCISYNC_ATTRIBUTES(int ,int)     = 106;*/

bool_t cuda_device_get_p2p_attribute_1_svc(int attr, int srcDevice, int dstDevice, int_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cudaDeviceGetP2PAttribute");
    result->err = cudaDeviceGetP2PAttribute(&result->int_result_u.data, attr,
                                            srcDevice, dstDevice);
    return 1;
}

bool_t cuda_device_get_pci_bus_id_1_svc(int len, int device, str_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cudaDeviceGetPCIBusId");
    if ((result->str_result_u.str = malloc(len)) == NULL) {
        LOGE(LOG_ERROR, "malloc failed");
        return 0;
    }

    result->err = cudaDeviceGetPCIBusId(result->str_result_u.str, len, device);
    return 1;
}

bool_t cuda_device_get_shared_mem_config_1_svc(int_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cudaDeviceGetSharedMemConfig");
    result->err = cudaDeviceGetSharedMemConfig((enum cudaSharedMemConfig*)&result->int_result_u.data);
    return 1;
}

bool_t cuda_device_get_stream_priority_range_1_svc(dint_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cudaDeviceGetStreamPriorityRange");
    result->err = cudaDeviceGetStreamPriorityRange(&result->dint_result_u.data.i1,                      &result->dint_result_u.data.i2);
    return 1;
}

bool_t cuda_device_get_texture_lmw_1_svc(cuda_channel_format_desc fmtDesc, int device, u64_result *result, struct svc_req *rqstp)
{
#if CUDART_VERSION >= 11000
    struct cudaChannelFormatDesc desc = {
        .f = fmtDesc.f,
        .w = fmtDesc.w,
        .x = fmtDesc.x,
        .y = fmtDesc.y,
        .z = fmtDesc.z,
    };
    LOGE(LOG_DEBUG, "cudaDeviceGetTexture1DLinearMaxWidth");
    result->err = cudaDeviceGetTexture1DLinearMaxWidth(&result->u64_result_u.u64,
                      &desc, device);
    return 1;
#else
    LOGE(LOG_ERROR, "not compiled with CUDA 11 support");
    return 1;
#endif
}

bool_t cuda_device_reset_1_svc(int *result, struct svc_req *rqstp)
{
    RECORD_VOID_API;
    LOGE(LOG_DEBUG, "cudaDeviceReset");
    *result = cudaDeviceReset();
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t cuda_device_set_cache_config_1_svc(int cacheConfig, int *result, struct svc_req *rqstp)
{
    RECORD_API(int);
    RECORD_SINGLE_ARG(cacheConfig);
    LOGE(LOG_DEBUG, "cudaFuncSetCacheConfig");
    *result = cudaDeviceSetCacheConfig(cacheConfig);
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t cuda_device_set_limit_1_svc(int limit, size_t value, int *result, struct svc_req *rqstp)
{
    RECORD_API(cuda_device_set_limit_1_argument);
    RECORD_ARG(1, limit);
    RECORD_ARG(2, value);
    LOGE(LOG_DEBUG, "cudaFuncSetLimit");
    *result = cudaDeviceSetLimit(limit, value);
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t cuda_device_set_shared_mem_config_1_svc(int config, int *result, struct svc_req *rqstp)
{
    RECORD_API(int);
    RECORD_SINGLE_ARG(config);
    LOGE(LOG_DEBUG, "cudaFuncSetSharedMemConfig");
    *result = cudaDeviceSetSharedMemConfig(config);
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t cuda_device_synchronize_1_svc(int *result, struct svc_req *rqstp)
{
    RECORD_VOID_API;
    LOGE(LOG_DEBUG, "cudaDeviceSynchronize");
    *result = cudaDeviceSynchronize();
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t cuda_get_device_1_svc(int_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cudaGetDevice");
    result->err = cudaGetDevice(&result->int_result_u.data);
    return 1;
}

bool_t cuda_get_device_count_1_svc(int_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cudaGetDeviceCount");
    result->err = cudaGetDeviceCount(&result->int_result_u.data);
    return 1;
}


bool_t cuda_get_device_flags_1_svc(int_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cudaGetDeviceFlags");
    result->err = cudaGetDeviceFlags((unsigned*)&result->int_result_u.data);
    return 1;
}

bool_t cuda_get_device_properties_1_svc(int device, mem_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cudaGetDeviceProperties");
    result->mem_result_u.data.mem_data_val = malloc(sizeof(struct cudaDeviceProp));
    if (result->mem_result_u.data.mem_data_val == NULL) {
        LOGE(LOG_ERROR, "malloc failed.");
        return 0;
    }
    result->mem_result_u.data.mem_data_len = sizeof(struct cudaDeviceProp);
    result->err = cudaGetDeviceProperties((void*)result->mem_result_u.data.mem_data_val, device);
    if (result->err != 0) {
        free(result->mem_result_u.data.mem_data_val);
    }
    return 1;
}

//        /*int        CUDA_IPC_CLOSE_MEM_HANDLE(ptr)                     = 121;*/
//        /*ptr_result CUDA_IPC_GET_EVENT_HANDLE(int)                     = 122;*/
//        /*ptr_result CUDA_IPC_GET_MEM_HANDLE(ptr)                       = 123;*/
//        /*ptr_result CUDA_IPC_OPEN_EVENT_HANDLE(ptr)                    = 124;*/
//        /*ptr_result CUDA_IPC_OPEN_MEM_HANDLE(ptr, int)                 = 125;*/

bool_t cuda_set_device_1_svc(int device, int *result, struct svc_req *rqstp)
{
    RECORD_API(int);
    RECORD_SINGLE_ARG(device);
    LOGE(LOG_DEBUG, "cudaSetDevice(%d)", device);
    *result = cudaSetDevice(device);
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t cuda_set_device_flags_1_svc(int flags, int *result, struct svc_req *rqstp)
{
    RECORD_API(int);
    RECORD_SINGLE_ARG(flags);
    LOGE(LOG_DEBUG, "cudaSetDevice");
    *result = cudaSetDeviceFlags(flags);
    RECORD_RESULT(integer, *result);
    return 1;
}

struct cuda_set_valid_device_param {
    int* arg1;
    int arg2;
};
bool_t cuda_set_valid_devices_1_svc(mem_data device_arr, int len, int *result, struct svc_req *rqstp)
{
    RECORD_API(struct cuda_set_valid_device_param);
#ifdef WITH_RECORDER
    int *valid_device = malloc(len*sizeof(int));
    if (valid_device == NULL) {
        LOGE(LOG_ERROR, "malloc failed.");
        return 0;
    }
    /* TODO: We actually create a memory leak here. We need to explicity
     * clean up the allocated memory region when the recorder list is cleaned */
    memcpy(valid_device, device_arr.mem_data_val, len*sizeof(int));
#endif
    RECORD_ARG(1, valid_device);
    RECORD_ARG(2, len);
    LOGE(LOG_DEBUG, "cudaSetValidDevices");
    if (device_arr.mem_data_len != len*sizeof(int)) {
        LOGE(LOG_ERROR, "mismatch between expected size (%d) and received size (%d)", len*sizeof(int), device_arr.mem_data_len);
        return 0;
    }
    *result = cudaSetValidDevices((int*)device_arr.mem_data_val, len);
    RECORD_RESULT(integer, *result);
    return 1;
}


/* ### Error Handling ### */

bool_t cuda_get_error_name_1_svc(int error, str_result *result, struct svc_req *rqstp)
{
    const char* str;
    result->str_result_u.str = malloc(128);
    LOGE(LOG_DEBUG, "cudaGetErrorName");
    str = cudaGetErrorName((cudaError_t)error);
    strncpy(result->str_result_u.str, str, 128);
    result->err = 0;
    return 1;
}

bool_t cuda_get_error_string_1_svc(int error, str_result *result, struct svc_req *rqstp)
{
    const char* str;
    result->str_result_u.str = malloc(128);
    LOGE(LOG_DEBUG, "cudaGetErrorString");
    str = cudaGetErrorString((cudaError_t)error);
    strncpy(result->str_result_u.str, str, 128);
    result->err = 0;
    return 1;
}

bool_t cuda_get_last_error_1_svc(int *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cudaGetLastError");
    *result = cudaGetLastError();
    return 1;
}

bool_t cuda_peek_at_last_error_1_svc(int *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cudaPeekAtLastError");
    *result = cudaPeekAtLastError();
    return 1;
}

/* ### Stream Management ### */

bool_t cuda_ctx_reset_persisting_l2cache_1_svc(int *result, struct svc_req *rqstp)
{
#if CUDART_VERSION >= 11000
    RECORD_VOID_API;
    LOGE(LOG_DEBUG, "cudaCtxResetPersistingL2Cache");
    *result = cudaCtxResetPersistingL2Cache();

    RECORD_RESULT(integer, *result);
#else
    LOGE(LOG_ERROR, "Compiled without CUDA 11 support");
#endif
    return 1;
}


/* Requires us do call a callback on the client side to make sense */
//        int          CUDA_STREAM_ADD_CALLBACK(ptr, ptr, mem_data, int)  = 251;

/* Requires unified memory OR attaching a shared memory region. */
//        int          CUDA_STREAM_ATTACH_MEM_ASYNC(ptr, ptr, size_t, int)= 252;

/* Requires Graph API to make sense */
//        int          CUDA_STREAM_BEGIN_CAPTURE(ptr, int)                = 253;

bool_t cuda_stream_copy_attributes_1_svc(ptr dst, ptr src, int *result, struct svc_req *rqstp)
{
#if CUDART_VERSION >= 11000
    RECORD_API(cuda_stream_copy_attributes_1_argument);
    RECORD_ARG(1, dst);
    RECORD_ARG(2, src);
    LOGE(LOG_DEBUG, "cudaStreamCopyAttributes");
    *result = cudaStreamCopyAttributes(resource_mg_get(&rm_streams, (void*)dst),
                                       resource_mg_get(&rm_streams, (void*)src));

    RECORD_RESULT(integer, *result);
    return 1;
#else
    LOGE(LOG_ERROR, "not compiled with CUDA 11 support");
    return 1;
#endif
}

bool_t cuda_stream_create_1_svc(ptr_result *result, struct svc_req *rqstp)
{
    RECORD_VOID_API;
    LOGE(LOG_DEBUG, "cudaStreamCreate");
    result->err = cudaStreamCreate((void*)&result->ptr_result_u.ptr);
    if (resource_mg_create(&rm_streams, (void*)result->ptr_result_u.ptr) != 0) {
        LOGE(LOG_ERROR, "error in resource manager");
    }
    RECORD_RESULT(ptr_result_u, *result);
    return 1;
}

bool_t cuda_stream_create_with_flags_1_svc(int flags, ptr_result *result, struct svc_req *rqstp)
{
    RECORD_API(int);
    RECORD_SINGLE_ARG(flags);
    LOGE(LOG_DEBUG, "cudaStreamCreateWithFlags");
    result->err = cudaStreamCreateWithFlags((void*)&result->ptr_result_u.ptr, flags);
    if (resource_mg_create(&rm_streams, (void*)result->ptr_result_u.ptr) != 0) {
        LOGE(LOG_ERROR, "error in resource manager");
    }
    LOG(LOG_DEBUG, "add to stream rm: %p", result->ptr_result_u.ptr);

    RECORD_RESULT(ptr_result_u, *result);
    return 1;
}

bool_t cuda_stream_create_with_priority_1_svc(int flags, int priority, ptr_result *result, struct svc_req *rqstp)
{
    RECORD_API(cuda_stream_create_with_priority_1_argument);
    RECORD_ARG(1, flags);
    RECORD_ARG(2, priority);

    LOGE(LOG_DEBUG, "cudaStreamCreateWithPriority");
    result->err = cudaStreamCreateWithPriority((void*)&result->ptr_result_u.ptr, flags, priority);
    if (resource_mg_create(&rm_streams, (void*)result->ptr_result_u.ptr) != 0) {
        LOGE(LOG_ERROR, "error in resource manager");
    }

    RECORD_RESULT(ptr_result_u, *result);
    return 1;
}

bool_t cuda_stream_destroy_1_svc(ptr stream, int *result, struct svc_req *rqstp)
{
    RECORD_API(ptr);
    RECORD_SINGLE_ARG(stream);
    LOGE(LOG_DEBUG, "cudaStreamDestroy");
    *result = cudaStreamDestroy(resource_mg_get(&rm_streams, (void*)stream));
    RECORD_RESULT(integer, *result);
    return 1;
}

/* Capture API does not make sense without graph API */
//        /*ptr_result   CUDA_STREAM_END_CAPTURE(ptr)                       = 259;*/
/* What datatypes are in the union cudaStreamAttrValue? */
//        /* ?         CUDA_STREAM_GET_ATTRIBUTE(ptr, int)                = 260;*/
/* Capture API does not make sense without graph API */
//        /* ?         CUDA_STREAM_GET_CAPTURE_INFO(ptr)                  = 261;*/


bool_t cuda_stream_get_flags_1_svc(ptr hStream, int_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cudaStreamGetFlags");
    result->err = cudaStreamGetFlags(resource_mg_get(&rm_streams, (void*)hStream),
                                     (unsigned*)&result->int_result_u.data);
    return 1;
}

bool_t cuda_stream_get_priority_1_svc(ptr hStream, int_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cudaStreamGetPriority");
    result->err = cudaStreamGetPriority(
      resource_mg_get(&rm_streams, (void*)hStream),
      &result->int_result_u.data);
    return 1;
}

/* Capture API does not make sense without graph API */
//        /* ?         CUDA_STREAM_IS_CAPTURING(ptr)                      = 264;*/

bool_t cuda_stream_query_1_svc(ptr hStream, int *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cudaStreamQuery");
    *result = cudaStreamQuery(
      resource_mg_get(&rm_streams, (void*)hStream));
    return 1;
}

/* What datatypes are in the union cudaStreamAttrValue? */
//        /*int          CUDA_STREAM_SET_ATTRIBUTE(ptr, int, ?)             = 266;*/

bool_t cuda_stream_synchronize_1_svc(ptr stream, int *result, struct svc_req *rqstp)
{
    RECORD_API(uint64_t);
    RECORD_SINGLE_ARG(stream);
    LOGE(LOG_DEBUG, "cudaStreamSynchronize");
    *result = cudaStreamSynchronize(
      resource_mg_get(&rm_streams, (void*)stream));
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t cuda_stream_wait_event_1_svc(ptr stream, ptr event, int flags, int *result, struct svc_req *rqstp)
{
    RECORD_API(cuda_stream_wait_event_1_argument);
    RECORD_ARG(1, stream);
    RECORD_ARG(2, event);
    RECORD_ARG(3, flags);
    LOGE(LOG_DEBUG, "cudaStreamWaitEvent");
    *result = cudaStreamWaitEvent(
      resource_mg_get(&rm_streams, (void*)stream),
      resource_mg_get(&rm_events, (void*)event),
      flags);
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t cuda_thread_exchange_stream_capture_mode_1_svc(int mode, int_result *result, struct svc_req *rqstp)
{
    RECORD_API(int);
    RECORD_SINGLE_ARG(mode);
    LOGE(LOG_DEBUG, "cudaThreadExchangeStreamCaptureMode");
    result->int_result_u.data = mode;
    result->err = cudaThreadExchangeStreamCaptureMode((void*)&result->int_result_u.data);
    RECORD_RESULT(integer, result->int_result_u.data);
    return 1;
}

/* ### Event Management ### */

bool_t cuda_event_create_1_svc(ptr_result *result, struct svc_req *rqstp)
{
    RECORD_VOID_API;
    LOGE(LOG_DEBUG, "cudaEventCreate");
    result->err = cudaEventCreate((struct CUevent_st**)&result->ptr_result_u.ptr);
    if (resource_mg_create(&rm_events, (void*)result->ptr_result_u.ptr) != 0) {
        LOGE(LOG_ERROR, "error in resource manager");
    }
    RECORD_RESULT(ptr_result_u, *result);
    return 1;
}

bool_t cuda_event_create_with_flags_1_svc(int flags, ptr_result *result, struct svc_req *rqstp)
{
    RECORD_API(int);
    RECORD_SINGLE_ARG(flags);
    LOGE(LOG_DEBUG, "cudaEventCreateWithFlags");
    result->err = cudaEventCreateWithFlags((struct CUevent_st**)&result->ptr_result_u.ptr, flags);
    if (resource_mg_create(&rm_events, (void*)result->ptr_result_u.ptr) != 0) {
        LOGE(LOG_ERROR, "error in resource manager");
    }
    RECORD_RESULT(ptr_result_u, *result);
    return 1;
}

bool_t cuda_event_destroy_1_svc(ptr event, int *result, struct svc_req *rqstp)
{
    RECORD_API(ptr);
    RECORD_SINGLE_ARG(event);
    LOGE(LOG_DEBUG, "cudaEventDestroy");
    *result = cudaEventDestroy(
      resource_mg_get(&rm_events, (void*)event));
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t cuda_event_elapsed_time_1_svc(ptr start, ptr end, float_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cudaEventElapsedTime");
    result->err = cudaEventElapsedTime(&result->float_result_u.data,
      resource_mg_get(&rm_events, (void*)start),
      resource_mg_get(&rm_events, (void*)end));
    return 1;
}

bool_t cuda_event_query_1_svc(ptr event, int *result, struct svc_req *rqstp)
{
    RECORD_API(ptr);
    RECORD_SINGLE_ARG(event);
    LOGE(LOG_DEBUG, "cudaEventQuery");
    *result = cudaEventQuery(
      resource_mg_get(&rm_events, (void*)event));
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t cuda_event_record_1_svc(ptr event, ptr stream, int *result, struct svc_req *rqstp)
{
    RECORD_API(cuda_event_record_1_argument);
    RECORD_ARG(1, event);
    RECORD_ARG(2, stream);
    LOGE(LOG_DEBUG, "cudaEventRecord");
    *result = cudaEventRecord(
      resource_mg_get(&rm_events, (void*)event),
      resource_mg_get(&rm_streams, (void*)stream));
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t cuda_event_record_with_flags_1_svc(ptr event, ptr stream, int flags, int *result, struct svc_req *rqstp)
{
#if CUDART_VERSION >= 11000
    RECORD_API(cuda_event_record_with_flags_1_argument);
    RECORD_ARG(1, event);
    RECORD_ARG(2, stream);
    RECORD_ARG(3, flags);
    LOGE(LOG_DEBUG, "cudaEventRecordWithFlags");
    *result = cudaEventRecordWithFlags(
      resource_mg_get(&rm_events, (void*)event),
      resource_mg_get(&rm_streams, (void*)stream),
      flags);
    RECORD_RESULT(integer, *result);
    return 1;
#else
    LOGE(LOG_ERROR, "compiled without CUDA 11 support");
    return 1;
#endif
}

bool_t cuda_event_synchronize_1_svc(ptr event, int *result, struct svc_req *rqstp)
{
    RECORD_API(ptr);
    RECORD_SINGLE_ARG(event);
    LOGE(LOG_DEBUG, "cudaEventSynchronize");
    *result = cudaEventSynchronize(
      resource_mg_get(&rm_events, (void*)event));
    RECORD_RESULT(integer, *result);
    return 1;
}

/* Execution Control */

bool_t cuda_func_get_attributes_1_svc(ptr func, mem_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cudaFuncGetAttributes");
    result->mem_result_u.data.mem_data_val =
        malloc(sizeof(struct cudaFuncAttributes));
    result->mem_result_u.data.mem_data_len = sizeof(struct cudaFuncAttributes);
    result->err = cudaFuncGetAttributes(
      (struct cudaFuncAttributes*) result->mem_result_u.data.mem_data_val,
      (void*)func);
    /* func is a pointer to program memory. It will be static across executions,
     * so we do not need a resource manager */
    return 1;
}

bool_t cuda_func_set_attributes_1_svc(ptr func, int attr, int value, int *result, struct svc_req *rqstp)
{
    RECORD_API(cuda_func_set_attributes_1_argument);
    RECORD_ARG(1, func);
    RECORD_ARG(2, attr);
    RECORD_ARG(3, value);
    LOGE(LOG_DEBUG, "cudaFuncSetAttributes");
    *result = cudaFuncSetAttribute((void*)func, attr, value);
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t cuda_func_set_cache_config_1_svc(ptr func, int cacheConfig, int *result, struct svc_req *rqstp)
{
    RECORD_API(cuda_func_set_cache_config_1_argument);
    RECORD_ARG(1, func);
    RECORD_ARG(2, cacheConfig);
    LOGE(LOG_DEBUG, "cudaFuncSetCacheConfig");
    *result = cudaFuncSetCacheConfig((void*)func, cacheConfig);
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t cuda_func_set_shared_mem_config_1_svc(ptr func, int config, int *result, struct svc_req *rqstp)
{
    RECORD_API(cuda_func_set_shared_mem_config_1_argument);
    RECORD_ARG(1, func);
    RECORD_ARG(2, config);
    LOGE(LOG_DEBUG, "cudaFuncSetSharedMemConfig");
    *result = cudaFuncSetSharedMemConfig((void*)func, config);
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t cuda_launch_cooperative_kernel_1_svc(ptr func, rpc_dim3 gridDim, rpc_dim3 blockDim, mem_data args, size_t sharedMem, ptr stream, int *result, struct svc_req *rqstp)
{
    RECORD_API(cuda_launch_cooperative_kernel_1_argument);
    RECORD_ARG(1, func);
    RECORD_ARG(2, gridDim);
    RECORD_ARG(3, blockDim);
    //TODO: Store parameters explicitly
    //RECORD_ARG(4, args);
    RECORD_ARG(5, sharedMem);
    RECORD_ARG(6, stream);
    dim3 cuda_gridDim = {gridDim.x, gridDim.y, gridDim.z};
    dim3 cuda_blockDim = {blockDim.x, blockDim.y, blockDim.z};
    void **cuda_args;
    uint16_t *arg_offsets;
    size_t param_num = *((size_t*)args.mem_data_val);
    arg_offsets = (uint16_t*)(args.mem_data_val+sizeof(size_t));
    cuda_args = malloc(param_num*sizeof(void*));
    for (size_t i = 0; i < param_num; ++i) {
        cuda_args[i] = args.mem_data_val+sizeof(size_t)+param_num*sizeof(uint16_t)+arg_offsets[i];
//    LOGE(LOG_DEBUG, "arg: %p (%d)\n", *(void**)cuda_args[i], *(int*)cuda_args[i]);
    }

    LOGE(LOG_DEBUG, "cudaLaunchCooperativeKernel(func=%p, gridDim=[%d,%d,%d], blockDim=[%d,%d,%d], args=%p, sharedMem=%d, stream=%p)", func, cuda_gridDim.x, cuda_gridDim.y, cuda_gridDim.z, cuda_blockDim.x, cuda_blockDim.y, cuda_blockDim.z, cuda_args, sharedMem, (void*)stream);

    *result = cudaLaunchCooperativeKernel(
      (void*)func,
      cuda_gridDim,
      cuda_blockDim,
      cuda_args,
      sharedMem,
      resource_mg_get(&rm_streams, (void*)stream));
    RECORD_RESULT(integer, *result);
    LOGE(LOG_DEBUG, "cudaLaunchCooperativeKernel result: %d", *result);
    return 1;
}

bool_t cuda_launch_cooperative_kernel_multi_device_1_svc(ptr func, rpc_dim3 gridDim, rpc_dim3 blockDim, mem_data args, size_t sharedMem, ptr stream, int numDevices, int flags, int *result, struct svc_req *rqstp)
{
    RECORD_API(cuda_launch_cooperative_kernel_multi_device_1_argument);
    RECORD_ARG(1, func);
    RECORD_ARG(2, gridDim);
    RECORD_ARG(3, blockDim);
    //TODO: Store parameters explicitly
    //RECORD_ARG(4, args);
    RECORD_ARG(5, sharedMem);
    RECORD_ARG(6, stream);
    RECORD_ARG(7, numDevices);
    RECORD_ARG(8, flags);
    dim3 cuda_gridDim = {gridDim.x, gridDim.y, gridDim.z};
    dim3 cuda_blockDim = {blockDim.x, blockDim.y, blockDim.z};
    void **cuda_args;
    uint16_t *arg_offsets;
    size_t param_num = *((size_t*)args.mem_data_val);
    struct cudaLaunchParams lp;
    arg_offsets = (uint16_t*)(args.mem_data_val+sizeof(size_t));
    cuda_args = malloc(param_num*sizeof(void*));
    for (size_t i = 0; i < param_num; ++i) {
        cuda_args[i] = args.mem_data_val+sizeof(size_t)+param_num*sizeof(uint16_t)+arg_offsets[i];
        //LOGE(LOG_DEBUG, "arg: %p (%d)\n", *(void**)cuda_args[i], *(int*)cuda_args[i]);
    }

    LOGE(LOG_DEBUG, "cudaLaunchCooperativeKernelMultiDevice(func=%p, gridDim=[%d,%d,%d], blockDim=[%d,%d,%d], args=%p, sharedMem=%d, stream=%p)", func, cuda_gridDim.x, cuda_gridDim.y, cuda_gridDim.z, cuda_blockDim.x, cuda_blockDim.y, cuda_blockDim.z, cuda_args, sharedMem, (void*)stream);
    lp.args = cuda_args;
    lp.blockDim = cuda_blockDim;
    lp.func = (void*)func;
    lp.gridDim = cuda_gridDim;
    lp.sharedMem = sharedMem;
    lp.stream = resource_mg_get(&rm_streams, (void*)stream);
    *result = cudaLaunchCooperativeKernelMultiDevice(&lp, numDevices, flags);
    RECORD_RESULT(integer, *result);
    LOGE(LOG_DEBUG, "cudaLaunchCooperativeKernelMultiDevice result: %d", *result);
    return 1;
}

/* This would require RPCs in the opposite direction.
 * __host__ cudaError_t cudaLaunchHostFunc ( cudaStream_t stream, cudaHostFn_t fn, void* userData )
 *   Enqueues a host function call in a stream.
 */

bool_t cuda_launch_kernel_1_svc(ptr func, rpc_dim3 gridDim, rpc_dim3 blockDim,
                                mem_data args, size_t sharedMem, ptr stream,
                                int *result, struct svc_req *rqstp)
{
    RECORD_API(cuda_launch_kernel_1_argument);
    RECORD_ARG(1, func);
    RECORD_ARG(2, gridDim);
    RECORD_ARG(3, blockDim);
    RECORD_DATA(args.mem_data_len, args.mem_data_val);
    RECORD_ARG(5, sharedMem);
    RECORD_ARG(6, stream);
    dim3 cuda_gridDim = {gridDim.x, gridDim.y, gridDim.z};
    dim3 cuda_blockDim = {blockDim.x, blockDim.y, blockDim.z};
    void **cuda_args;
    uint16_t *arg_offsets;
    size_t param_num = *((size_t*)args.mem_data_val);
    arg_offsets = (uint16_t*)(args.mem_data_val+sizeof(size_t));
    cuda_args = malloc(param_num*sizeof(void*));
    for (size_t i = 0; i < param_num; ++i) {
        cuda_args[i] = args.mem_data_val+sizeof(size_t)+param_num*sizeof(uint16_t)+arg_offsets[i];
        *(void**)cuda_args[i] = resource_mg_get(&rm_memory, *(void**)cuda_args[i]);
        LOGE(LOG_DEBUG, "arg: %p (%d)", *(void**)cuda_args[i], *(int*)cuda_args[i]);
    }

    LOGE(LOG_DEBUG, "cudaLaunchKernel(func=%p, gridDim=[%d,%d,%d], blockDim=[%d,%d,%d], args=%p, sharedMem=%d, stream=%p)", func, cuda_gridDim.x, cuda_gridDim.y, cuda_gridDim.z, cuda_blockDim.x, cuda_blockDim.y, cuda_blockDim.z, cuda_args, sharedMem, (void*)stream);

    *result = cudaLaunchKernel(
      (void*)func,
      cuda_gridDim,
      cuda_blockDim,
      cuda_args,
      sharedMem,
      resource_mg_get(&rm_streams, (void*)stream));
    free(cuda_args);
    RECORD_RESULT(integer, *result);
    LOGE(LOG_DEBUG, "cudaLaunchKernel result: %d", *result);
    return 1;
}


/* cudaSetDoubleForDevice ( double* d ) is deprecated */
/* cudaSetDoubleForHost ( double* d ) is deprecated */

/* Occupancy */

bool_t cuda_occupancy_available_dsmpb_1_svc(ptr func, int numBlocks, int blockSize, u64_result *result, struct svc_req *rqstp)
{
#if CUDART_VERSION >= 11000
    LOGE(LOG_DEBUG, "cudaOccupancyAvailableDynamicSMemPerBlock");
    result->err = cudaOccupancyAvailableDynamicSMemPerBlock(
        &result->u64_result_u.u64, (void*)func, numBlocks, blockSize);
    return 1;
#else
    LOGE(LOG_ERROR, "compiled without CUDA 11 support");
    return 1;
#endif
}

bool_t cuda_occupancy_max_active_bpm_1_svc(ptr func, int blockSize, size_t dynamicSMemSize, int_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cudaOccupancyMaxActiveBlocksPerMultiprocessor");
    result->err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &result->int_result_u.data, (void*)func, blockSize, dynamicSMemSize);
    return 1;
}

bool_t cuda_occupancy_max_active_bpm_with_flags_1_svc(ptr func, int blockSize, size_t dynamicSMemSize, int flags, int_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags");
    result->err = cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        &result->int_result_u.data, (void*)func, blockSize, dynamicSMemSize, flags);
    return 1;
}

/* Memory Management */

bool_t cuda_array_get_info_1_svc(ptr array, mem_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cudaArrayGetInfo");
    result->mem_result_u.data.mem_data_len =
      sizeof(struct cudaChannelFormatDesc)+sizeof(struct cudaExtent)+sizeof(int);
    result->mem_result_u.data.mem_data_val = malloc(
      sizeof(struct cudaChannelFormatDesc)+sizeof(struct cudaExtent)+sizeof(int));
    struct cudaChannelFormatDesc* desc = (void*)result->mem_result_u.data.mem_data_val;
    struct cudaExtent *extent = (void*)result->mem_result_u.data.mem_data_val+
                                sizeof(struct cudaChannelFormatDesc);
    unsigned *flags = (void*)&result->mem_result_u.data.mem_data_val+
                      sizeof(struct cudaChannelFormatDesc)+
                      sizeof(struct cudaExtent);

    result->err = cudaArrayGetInfo(desc,
                                   extent,
                                   flags,
                                   resource_mg_get(&rm_arrays, (void*)array));
    return 1;
}

bool_t cuda_array_get_sparse_properties_1_svc(ptr array, mem_result *result, struct svc_req *rqstp)
{
#if CUDART_VERSION >= 11000
    LOGE(LOG_DEBUG, "cudaArrayGetSparseProperties");
    result->mem_result_u.data.mem_data_len =
      sizeof(struct cudaArraySparseProperties);
    result->mem_result_u.data.mem_data_val = malloc(
      sizeof(struct cudaArraySparseProperties));
    struct cudaArraySparseProperties* sparseProperties = (void*)result->mem_result_u.data.mem_data_val;

    result->err = cudaArrayGetSparseProperties(
      sparseProperties,
      resource_mg_get(&rm_arrays, (void*)array));
    return 1;
#else
    LOGE(LOG_ERROR, "not compiled with CUDA 11 support");
    return 1;
#endif
}

bool_t cuda_free_1_svc(ptr devPtr, int *result, struct svc_req *rqstp)
{
    RECORD_API(ptr);
    RECORD_SINGLE_ARG(devPtr);
    int index = hainfo_getserverindex((void*)devPtr);
    int ret = 1;
    uint64_t arg;
    api_record_t *r;
    LOGE(LOG_DEBUG, "cudaFree");

    #ifdef WITH_IB

    ib_free_memreg((void*)devPtr, index, true);
    hainfo[index].server_ptr = 0;
    *result = 0;

    #else

    *result = cudaFree(resource_mg_get(&rm_memory, (void*)devPtr));

    #endif

    /* The cleanup/simplification of the record could also be
     * done during checkpoint creation. What is better depends
     * on whether we want to minimize overhead of cudaFree or
     * of the checkpoint creation. */
    //for (size_t i = 0; i < api_records.length; ++i) {
    //    r = (api_record_t*)api_records.elements[i];
    //    switch (r->function) {
    //    case CUDA_MALLOC:
    //        arg = r->result.u64;
    //        break;
    //    case CUDA_MEMCPY_DTOD:
    //        arg = ((cuda_memcpy_dtod_1_argument*)r->arguments)->arg1;
    //        break;
    //    case CUDA_MEMCPY_HTOD:
    //        arg = ((cuda_memcpy_htod_1_argument*)r->arguments)->arg1;
    //        break;
    //    case CUDA_MEMCPY_TO_SYMBOL:
    //        arg = ((cuda_memcpy_to_symbol_1_argument*)r->arguments)->arg1;
    //        break;
    //    case CUDA_MEMCPY_TO_SYMBOL_SHM:
    //        arg = ((cuda_memcpy_to_symbol_shm_1_argument*)r->arguments)->arg2;
    //        break;
    //    case CUDA_MEMCPY_IB:
    //        arg = ((cuda_memcpy_ib_1_argument*)r->arguments)->arg2;
    //        break;
    //    case CUDA_MEMCPY_SHM:
    //        arg = ((cuda_memcpy_shm_1_argument*)r->arguments)->arg2;
    //        break;
    //    default:
    //        continue;
    //    }
    //    if (arg == devPtr) {
    //        list_rm(&api_records, i, NULL);
    //        --i;
    //        ret = 0;
    //    }
    //}
    //if (ret != 0) {
    //    LOGE(LOG_ERROR, "could not find a malloc call associated with this free call");
    //    *result = CUDA_ERROR_UNKNOWN;
    //}
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t cuda_free_array_1_svc(ptr devPtr, int *result, struct svc_req *rqstp)
{
    RECORD_API(ptr);
    RECORD_SINGLE_ARG(devPtr);
    LOGE(LOG_DEBUG, "cudaFreeArray");
    *result = cudaFreeArray(resource_mg_get(&rm_arrays, (void*)devPtr));
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t cuda_free_host_1_svc(int index, int *result, struct svc_req *rqstp)
{
    //TODO: Do we need to use a resource manager here? Not sure yet.
    RECORD_API(int);
    RECORD_SINGLE_ARG(index);
    *result = cudaErrorInitializationError;
    if (socktype != UNIX) {
        *result = cudaSuccess;
        return 1;
    }
    if (hainfo[index].cnt != 0 &&
        hainfo[index].cnt == index) {

        *result = cudaHostUnregister(hainfo[index].server_ptr);
        munmap(hainfo[index].server_ptr, hainfo[index].size);
        memset(&hainfo[index], 0, sizeof(host_alloc_info_t));
        if (*result == cudaErrorHostMemoryNotRegistered) {
            *result = cudaSuccess;
        }
    }
    RECORD_RESULT(integer, *result);
    return 1;
}

/* cudaFreeMipmappedArray ( cudaMipmappedArray_t mipmappedArray ) is not implemented */
/* cudaGetMipmappedArrayLevel ( cudaArray_t* levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned int  level ) is not implemented */


bool_t cuda_get_symbol_address_1_svc(ptr symbol, ptr_result *result, struct svc_req *rqstp)
{
    //Symbol is in host address space
    //TODO: Does the returned device address space address change across
    // executions? If yes we need to manage it using a device manager.
    LOGE(LOG_DEBUG, "cudaGetSymbolAddress");
    result->err = cudaGetSymbolAddress((void**)&result->ptr_result_u.ptr, (void*)symbol);
    return 1;
}

bool_t cuda_get_symbol_size_1_svc(ptr symbol, u64_result *result, struct svc_req *rqstp)
{
    //Symbol is in host address space
    LOGE(LOG_DEBUG, "cudaGetSymbolSize");
    result->err = cudaGetSymbolSize(&result->u64_result_u.u64, (void*)symbol);
    return 1;
}

bool_t cuda_host_alloc_1_svc(int client_cnt, size_t size, ptr client_ptr, unsigned int flags, int *result, struct svc_req *rqstp)
{
    //TODO: Make checkpointable. Implement reattaching of shm segment.
    int fd_shm;
    char shm_name[128];
    void *shm_addr;
    unsigned int register_flags = 0;
    *result = cudaErrorMemoryAllocation;
    RECORD_API(cuda_host_alloc_1_argument);
    RECORD_ARG(1, client_cnt);
    RECORD_ARG(2, size);
    RECORD_ARG(3, client_ptr);
    RECORD_ARG(4, flags);

    LOGE(LOG_DEBUG, "cudaHostAlloc");

    if (socktype == UNIX || (shm_enabled && cpu_utils_is_local_connection(rqstp))) { //Use local shared memory
        snprintf(shm_name, 128, "/crickethostalloc-%d", client_cnt);
        if ((fd_shm = shm_open(shm_name, O_RDWR, 600)) == -1) {
            LOGE(LOG_ERROR, "could not open shared memory \"%s\" with size %d: %s", shm_name, size, strerror(errno));
            goto out;
        }
        if ((shm_addr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_shm, 0)) == MAP_FAILED) {
            LOGE(LOG_ERROR, "mmap returned unexpected pointer: %p", shm_addr);
            goto cleanup;
        }

        if (flags & cudaHostAllocPortable) {
            register_flags |= cudaHostRegisterPortable;
        }
        if (flags & cudaHostAllocMapped) {
            register_flags |= cudaHostRegisterMapped;
        }
        if (flags & cudaHostAllocWriteCombined) {
            register_flags |= cudaHostRegisterMapped;
        }

        if ((*result = cudaHostRegister(shm_addr, size, flags)) != cudaSuccess) {
            LOGE(LOG_ERROR, "cudaHostRegister failed.");
            munmap(shm_addr, size);
            goto cleanup;
        }

        hainfo[hainfo_cnt].cnt = client_cnt;
        hainfo[hainfo_cnt].size = size;
        hainfo[hainfo_cnt].client_ptr = (void*)client_ptr;
        hainfo[hainfo_cnt].server_ptr = shm_addr;
        hainfo_cnt++;
    } else if (socktype == TCP) { //Use infiniband
#ifdef WITH_IB
   
#else
                LOGE(LOG_ERROR, "infiniband is disabled.");
                goto cleanup;
#endif //WITH_IB

    } else {
        LOGE(LOG_ERROR, "cudaHostAlloc is not supported for other transports than UNIX. This error means cricket_server and cricket_client are not compiled correctly (different transports)");
        goto out;
    }

    *result = cudaSuccess;
cleanup:
    shm_unlink(shm_name);
out:
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t cuda_host_get_device_pointer_1_svc(ptr pHost, int flags, ptr_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cudaHostGetDevicePointer");
    //TODO: implement correctly using the resouce manager
    //result->err = cudaHostGetDevicePointer((void**)&result->ptr_result_u.ptr, (void*)pHost, flags);
    result->err = cudaErrorUnknown;
    return 1;
}

bool_t cuda_host_get_flags_1_svc(ptr pHost, int_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cudaHostGetFlags");
    //TODO: implement correctly using the resouce manager
    //result->err = cudaHostFlags(&result->int_result_u.data, (void*)pHost);
    result->err = cudaErrorUnknown;
    return 1;
}
/* cudaHostRegister ( void* ptr, size_t size, unsigned int  flags ) not possible using shared memory
 * maybe we can register infiniband segment though?
 */
/* cudaHostUnregister ( void* ptr ) same as above */
//here we will register new ib region 
bool_t cuda_malloc_1_svc(size_t argp, ptr_result *result, struct svc_req *rqstp)
{   
    RECORD_API(size_t);
    RECORD_SINGLE_ARG(argp);
    LOGE(LOG_DEBUG, "cudaMalloc");


#ifdef WITH_IB
        result->err = ib_allocate_memreg((void**)&result->ptr_result_u.ptr, argp, hainfo_cnt, true);
            if (result->err == 0) {
                hainfo[hainfo_cnt].cnt = hainfo_cnt;
                hainfo[hainfo_cnt].size = argp;
                hainfo[hainfo_cnt].server_ptr = (void*)result->ptr_result_u.ptr;

                hainfo_cnt++;
            }    
#else
    result->err = cudaMalloc((void **)&result->ptr_result_u.ptr, argp);
    resource_mg_create(&rm_memory, (void *)result->ptr_result_u.ptr);
#endif

    RECORD_RESULT(ptr_result_u, *result);
    return 1;
}

bool_t cuda_malloc_3d_1_svc(size_t depth, size_t height, size_t width, pptr_result *result, struct svc_req *rqstp)
{
    RECORD_API(cuda_malloc_3d_1_argument);
    RECORD_ARG(1, depth);
    RECORD_ARG(2, height);
    RECORD_ARG(3, width);
    struct cudaExtent extent = {.depth = depth,
                                .height = height,
                                .width = width};
    struct cudaPitchedPtr pptr;
    LOGE(LOG_DEBUG, "cudaMalloc3D");
    result->err = cudaMalloc3D(&pptr, extent);
    result->pptr_result_u.ptr.pitch = pptr.pitch;
    result->pptr_result_u.ptr.ptr = (ptr)pptr.ptr;
    result->pptr_result_u.ptr.xsize = pptr.xsize;
    result->pptr_result_u.ptr.ysize = pptr.ysize;
    resource_mg_create(&rm_memory, pptr.ptr);

    RECORD_RESULT(integer, result->err);
    return 1;
}

bool_t cuda_malloc_3d_array_1_svc(cuda_channel_format_desc desc, size_t depth, size_t height, size_t width, int flags, ptr_result *result, struct svc_req *rqstp)
{
    RECORD_API(cuda_malloc_3d_array_1_argument);
    RECORD_ARG(1, desc);
    RECORD_ARG(2, depth);
    RECORD_ARG(3, height);
    RECORD_ARG(4, width);
    RECORD_ARG(5, flags);
    struct cudaChannelFormatDesc cuda_desc = {
      .f = desc.f,
      .w = desc.w,
      .x = desc.x,
      .y = desc.y,
      .z = desc.z};
    struct cudaExtent extent = {.depth = depth,
                                .height = height,
                                .width = width};
    LOGE(LOG_DEBUG, "cudaMalloc3DArray");
    result->err = cudaMalloc3DArray((void*)&result->ptr_result_u.ptr, &cuda_desc, extent, flags);
    resource_mg_create(&rm_arrays, (void*)result->ptr_result_u.ptr);

    RECORD_RESULT(integer, result->err);
    return 1;
}

bool_t cuda_malloc_array_1_svc(cuda_channel_format_desc desc, size_t width, size_t height, int flags, ptr_result *result, struct svc_req *rqstp)
{
    RECORD_API(cuda_malloc_array_1_argument);
    RECORD_ARG(1, desc);
    RECORD_ARG(2, width);
    RECORD_ARG(3, height);
    RECORD_ARG(4, flags);
    struct cudaChannelFormatDesc cuda_desc = {
      .f = desc.f,
      .w = desc.w,
      .x = desc.x,
      .y = desc.y,
      .z = desc.z};
    LOGE(LOG_DEBUG, "cudaMallocArray");
    result->err = cudaMallocArray((void*)&result->ptr_result_u.ptr, &cuda_desc, width, height, flags);
    resource_mg_create(&rm_arrays, (void*)result->ptr_result_u.ptr);

    RECORD_RESULT(integer, result->err);
    return 1;
}

/* cudaMallocHost ( void** ptr, size_t size ) is not implemented */
/* cudaMallocManaged ( void** devPtr, size_t size, unsigned int  flags = cudaMemAttachGlobal ) is not implemented */
/* cudaMallocMipmappedArray ( cudaMipmappedArray_t* mipmappedArray, const cudaChannelFormatDesc* desc, cudaExtent extent, unsigned int  numLevels, unsigned int  flags = 0 ) is not implemented */

bool_t cuda_malloc_pitch_1_svc(size_t width, size_t height, ptrsz_result *result, struct svc_req *rqstp)
{
    RECORD_API(cuda_malloc_pitch_1_argument);
    RECORD_ARG(1, width);
    RECORD_ARG(2, height);
    LOGE(LOG_DEBUG, "cudaMallocPitch");
    result->err = cudaMallocPitch((void*)&result->ptrsz_result_u.data.p,
                                  &result->ptrsz_result_u.data.s,
                                  width, height);
    resource_mg_create(&rm_memory, (void*)result->ptrsz_result_u.data.p);

    RECORD_RESULT(integer, result->err);
    return 1;
}

bool_t cuda_mem_advise_1_svc(ptr devPtr, size_t count, int advice, int device, int *result, struct svc_req *rqstp)
{
    RECORD_API(cuda_mem_advise_1_argument);
    RECORD_ARG(1, devPtr);
    RECORD_ARG(2, count);
    RECORD_ARG(3, advice);
    RECORD_ARG(4, device);

    LOGE(LOG_DEBUG, "cudaMemAdvise");
    *result = cudaMemAdvise(
      resource_mg_get(&rm_memory, (void*)devPtr),
      count, advice, device);

    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t cuda_mem_get_info_1_svc(dsz_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cudaMemGetInfo");
    result->err = cudaMemGetInfo(&result->dsz_result_u.data.sz1,
                             &result->dsz_result_u.data.sz2);
    return 1;
}

bool_t cuda_mem_prefetch_async_1_svc(ptr devPtr, size_t count, int dstDevice, ptr stream, int *result, struct svc_req *rqstp)
{
    RECORD_API(cuda_mem_prefetch_async_1_argument);
    RECORD_ARG(1, devPtr);
    RECORD_ARG(2, count);
    RECORD_ARG(3, dstDevice);
    RECORD_ARG(4, stream);

    LOGE(LOG_DEBUG, "cudaMemPrefetchAsync");
    *result = cudaMemPrefetchAsync(
      resource_mg_get(&rm_memory, (void*)devPtr),
      count, dstDevice, (void*)stream);

    RECORD_RESULT(integer, *result);
    return 1;
}

/* cuda_mem_range_get_attribute_1_svc(int attribute, ptr devPtr, size_t count, mem_result *result, struct svc_req *rqstp) unsupported because requires unified memors */
/* cudaMemRangeGetAttribute ( void* data, size_t dataSize, cudaMemRangeAttribute attribute, const void* devPtr, size_t count ) unsupported because requires unified memory */

/* CUDA_MEMCPY Family */

bool_t cuda_memcpy_htod_1_svc(uint64_t ptr, mem_data mem, size_t size, int *result, struct svc_req *rqstp)
{
    RECORD_API(cuda_memcpy_htod_1_argument);
    RECORD_ARG(1, ptr);
    RECORD_ARG(2, mem);
    RECORD_ARG(3, size);

    LOGE(LOG_DEBUG, "cudaMemcpyHtoD");
    if (size != mem.mem_data_len) {
        LOGE(LOG_ERROR, "data size mismatch");
        *result = cudaErrorUnknown;
        return 1;
    }
#ifdef WITH_MEMCPY_REGISTER
    if ((*result = cudaHostRegister(mem.mem_data_val, size, cudaHostRegisterMapped)) != cudaSuccess) {
        LOGE(LOG_ERROR, "cudaHostRegister failed: %d.", *result);
        return 1;
    }
#endif
    *result = cudaMemcpy(
      resource_mg_get(&rm_memory, (void*)ptr),
      mem.mem_data_val,
      size,
      cudaMemcpyHostToDevice);
#ifdef WITH_MEMCPY_REGISTER
    cudaHostUnregister(mem.mem_data_val);
#endif

    RECORD_RESULT(integer, *result);
    return 1;
}


bool_t cuda_memcpy_dtod_1_svc(ptr dst, ptr src, size_t size, int *result, struct svc_req *rqstp)
{
    RECORD_API(cuda_memcpy_dtod_1_argument);
    RECORD_ARG(1, dst);
    RECORD_ARG(2, src);
    RECORD_ARG(3, size);

    LOGE(LOG_DEBUG, "cudaMemcpyDtoD");
    *result = cudaMemcpy(
      resource_mg_get(&rm_memory, (void*)dst),
      resource_mg_get(&rm_memory, (void*)src),
      size, cudaMemcpyDeviceToDevice);

    RECORD_RESULT(integer, *result);
    return 1;
}


#ifdef WITH_IB
struct ib_thread_info {
    int index;
    void* host_ptr;
    void *device_ptr;
    size_t size;
    int result;
};
//is thread needed?
void* ib_thread(void* arg)
{
    struct ib_thread_info *info = (struct ib_thread_info*)arg;
    info->result = 0;
    ib_responder_recv(info->host_ptr, info->index, info->size, true);

    free (info);
    return NULL;
}
//the device ptr points to used device mem as is (as void*)
bool_t cuda_memcpy_ib_1_svc(int index, ptr device_ptr, size_t size, int kind, int *result, struct svc_req *rqstp)
{
    index = hainfo_getserverindex((void*)device_ptr);
    RECORD_API(cuda_memcpy_ib_1_argument);
    RECORD_ARG(1, index);
    RECORD_ARG(2, device_ptr);
    RECORD_ARG(3, size);
    RECORD_ARG(4, kind);
    LOGE(LOG_DEBUG, "cudaMemcpyIB");
    *result = cudaErrorInitializationError;
    //anstatt array list (list.c)
    if (hainfo[index].cnt == 0 ||
        hainfo[index].cnt != index) {

        LOGE(LOG_ERROR, "inconsistent state");
        goto out;
    }
    if (hainfo[index].size < size) {
        LOGE(LOG_ERROR, "requested size is smaller than ib memory segment");
        goto out;
    }

    if (kind == cudaMemcpyHostToDevice) {
        pthread_t thread = {0};
        // host ptr is device pointer!
        struct ib_thread_info *info = malloc(sizeof(struct ib_thread_info));
        info->index = index;
        info->host_ptr = hainfo[index].server_ptr;
//        info->device_ptr = resource_mg_get(&rm_memory, (void*)device_ptr);
        info->size = size;
        info->result = 0;
        if (pthread_create(&thread, NULL, ib_thread, info) != 0) {
            LOGE(LOG_ERROR, "starting ib thread failed.");
            goto out;
        }
        *result = cudaSuccess;

    } else if (kind == cudaMemcpyDeviceToHost) {

          *result = 0;
          ib_requester_send(hainfo[index].server_ptr, index, size, true);

        //TODO: Replace hardcoded IB destination below (Environment variable?) -> DONE
        //first arg will bei ib registered device mem reg -> ?
    }
out:
    RECORD_RESULT(integer, *result);
    return 1;
}
#else
bool_t cuda_memcpy_ib_1_svc(int index, ptr device_ptr, size_t size, int kind, int *result, struct svc_req *rqstp)
{
    LOGE(LOG_ERROR, "compiled without infiniband support");
    return 1;
}
#endif //WITH_IB

bool_t cuda_memcpy_shm_1_svc(int index, ptr device_ptr, size_t size, int kind, int *result, struct svc_req *rqstp)
{
    RECORD_API(cuda_memcpy_shm_1_argument);
    RECORD_ARG(1, index);
    RECORD_ARG(2, device_ptr);
    RECORD_ARG(3, size);
    RECORD_ARG(4, kind);
    LOGE(LOG_DEBUG, "cudaMemcpyShm");
    *result = cudaErrorInitializationError;
    if (hainfo[index].cnt == 0 ||
        hainfo[index].cnt != index) {

        LOGE(LOG_ERROR, "inconsistent state");
        goto out;
    }
    if (hainfo[index].size < size) {
        LOGE(LOG_ERROR, "requested size is smaller than shared memory segment");
        goto out;
    }

    if (kind == cudaMemcpyHostToDevice) {
        *result = cudaMemcpy(
          resource_mg_get(&rm_memory, (void*)device_ptr),
          hainfo[index].server_ptr,
          size,
          kind);
    } else if (kind == cudaMemcpyDeviceToHost) {
        *result = cudaMemcpy(
          hainfo[index].server_ptr,
          resource_mg_get(&rm_memory, (void*)device_ptr),
          size,
          kind);
    }
out:
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t cuda_memcpy_dtoh_1_svc(uint64_t ptr, size_t size, mem_result *result, struct svc_req *rqstp)
{
    //Does not need to be recorded because doesn't change device state
    LOGE(LOG_DEBUG, "cudaMemcpyDtoH(%p, %zu)", ptr, size);
    result->mem_result_u.data.mem_data_len = size;
    result->mem_result_u.data.mem_data_val = malloc(size);
#ifdef WITH_MEMCPY_REGISTER
    if ((result->err = cudaHostRegister(result->mem_result_u.data.mem_data_val,
                                        size, cudaHostRegisterMapped)) != cudaSuccess) {
        LOGE(LOG_ERROR, "cudaHostRegister failed.");
        goto out;
    }
#endif
    result->err = cudaMemcpy(
      result->mem_result_u.data.mem_data_val,
      resource_mg_get(&rm_memory, (void*)ptr),
      size,
      cudaMemcpyDeviceToHost);
#ifdef WITH_MEMCPY_REGISTER
    cudaHostUnregister(result->mem_result_u.data.mem_data_val);
#endif
    if (result->err != 0) {
        free(result->mem_result_u.data.mem_data_val);
    }
out:
    return 1;
}

/* Multidimensional Memcpys */

/* cudaMemcpy2D ( void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind ) not implemented yet */
/* cudaMemcpy2DArrayToArray ( cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, cudaMemcpyKind kind = cudaMemcpyDeviceToDevice ) not implemented yet */
/* cudaMemcpy2DAsync ( void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0 ) not implemented yet */
/* cudaMemcpy2DFromArray ( void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind ) not implemented yet */
/* cudaMemcpy2DFromArrayAsync ( void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0 ) not implemented yet */
/* cudaMemcpy2DToArray ( cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind ) not implemented yet */
/* cudaMemcpy2DToArrayAsync ( cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0 ) not implemented yet */
/* cudaMemcpy3D ( const cudaMemcpy3DParms* p ) not implemented yet */
/* cudaMemcpy3DAsync ( const cudaMemcpy3DParms* p, cudaStream_t stream = 0 ) not implemented yet */
/* cudaMemcpy3DPeer ( const cudaMemcpy3DPeerParms* p ) not implemented yet */
/* cudaMemcpy3DPeerAsync ( const cudaMemcpy3DPeerParms* p, cudaStream_t stream = 0 ) not implemented yet */

/* More memcpy Family */

/* cudaMemcpyAsync ( void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0 ) not implemented yet */
/* cudaMemcpyFromSymbol ( void* dst, const void* symbol, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyDeviceToHost ) not implemented yet. see cudaMemcpyToSymbol */
/* cudaMemcpyFromSymbolAsync ( void* dst, const void* symbol, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream = 0 ) not implemented yet */
/* cudaMemcpyPeer ( void* dst, int  dstDevice, const void* src, int  srcDevice, size_t count ) not implemented yet. see cudaMemcpyDtoD */
/* cudaMemcpyPeerAsync ( void* dst, int  dstDevice, const void* src, int  srcDevice, size_t count, cudaStream_t stream = 0 ) */

bool_t cuda_memcpy_to_symbol_1_svc(uint64_t ptr, mem_data mem, size_t size, size_t offset, int *result, struct svc_req *rqstp)
{
    RECORD_API(cuda_memcpy_to_symbol_1_argument);
    RECORD_ARG(1, ptr);
    RECORD_ARG(2, mem);
    RECORD_ARG(3, size);
    RECORD_ARG(4, offset);

    LOGE(LOG_DEBUG, "cudaMemcpyToSymbol");
    if (size != mem.mem_data_len) {
        LOGE(LOG_ERROR, "data size mismatch");
        *result = cudaErrorUnknown;
        return 1;
    }
#ifdef WITH_MEMCPY_REGISTER
    if ((*result = cudaHostRegister(mem.mem_data_val, size, cudaHostRegisterMapped)) != cudaSuccess) {
        LOGE(LOG_ERROR, "cudaHostRegister failed: %d.", *result);
        return 1;
    }
#endif
    *result = cudaMemcpyToSymbol((void*)ptr, mem.mem_data_val, size, offset, cudaMemcpyHostToDevice);
#ifdef WITH_MEMCPY_REGISTER
    cudaHostUnregister(mem.mem_data_val);
#endif
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t cuda_memcpy_to_symbol_shm_1_svc(int index, ptr device_ptr, size_t size, size_t offset, int kind, int *result, struct svc_req *rqstp)
{
    RECORD_API(cuda_memcpy_to_symbol_shm_1_argument);
    RECORD_ARG(1, index);
    RECORD_ARG(2, device_ptr);
    RECORD_ARG(3, size);
    RECORD_ARG(4, offset);
    RECORD_ARG(5, kind);
    LOGE(LOG_DEBUG, "cudaMemcpyToSymbolShm");
    *result = cudaErrorInitializationError;
    if (hainfo[index].cnt == 0 ||
        hainfo[index].cnt != index) {

        LOGE(LOG_ERROR, "inconsistent state");
        goto out;
    }
    if (hainfo[index].size < size) {
        LOGE(LOG_ERROR, "requested size is smaller than shared memory segment");
        goto out;
    }

    if (kind == cudaMemcpyHostToDevice) {
        *result = cudaMemcpyToSymbol((void*)device_ptr, hainfo[index].server_ptr, size, offset, kind);
    } else {
        LOGE(LOG_ERROR, "a kind different from HostToDevice is unsupported for cudaMemcpyToSymbol");
    }
out:
    RECORD_RESULT(integer, *result);
    return 1;
}

/* cudaMemcpyToSymbolAsync ( const void* symbol, const void* src, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream = 0 ) not implemented yet */

/* cudaMemset family */
bool_t cuda_memset_1_svc(ptr devPtr, int value, size_t count, int *result, struct svc_req *rqstp)
{
    RECORD_API(cuda_memset_1_argument);
    RECORD_ARG(1, devPtr);
    RECORD_ARG(2, value);
    RECORD_ARG(3, count);
    LOGE(LOG_DEBUG, "cudaMemset");
    *result = cudaMemset(
      resource_mg_get(&rm_memory, (void*)devPtr),
      value,
      count);
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t cuda_memset_2d_1_svc(ptr devPtr, size_t pitch, int value, size_t width, size_t height, int *result, struct svc_req *rqstp)
{
    RECORD_API(cuda_memset_2d_1_argument);
    RECORD_ARG(1, devPtr);
    RECORD_ARG(2, pitch);
    RECORD_ARG(3, value);
    RECORD_ARG(4, height);
    RECORD_ARG(5, width);
    LOGE(LOG_DEBUG, "cudaMemset2D");
    *result = cudaMemset2D(
      resource_mg_get(&rm_memory, (void*)devPtr),
      pitch,
      value,
      width,
      height);
    RECORD_RESULT(integer, *result);
    return 1;
}

/* cudaMemset2DAsync ( void* devPtr, size_t pitch, int  value, size_t width, size_t height, cudaStream_t stream = 0 ) is not implemented */

bool_t cuda_memset_3d_1_svc(size_t pitch, ptr devPtr, size_t xsize, size_t ysize, int value, size_t depth, size_t height, size_t width, int *result, struct svc_req *rqstp)
{
    RECORD_API(cuda_memset_3d_1_argument);
    RECORD_ARG(1, pitch);
    RECORD_ARG(2, devPtr);
    RECORD_ARG(3, xsize);
    RECORD_ARG(4, ysize);
    RECORD_ARG(5, value);
    RECORD_ARG(6, depth);
    RECORD_ARG(7, height);
    RECORD_ARG(8, width);
    LOGE(LOG_DEBUG, "cudaMemset3D");
    struct cudaPitchedPtr pptr = {.pitch = pitch,
                                  .ptr = resource_mg_get(&rm_memory, (void*)devPtr),
                                  .xsize = xsize,
                                  .ysize = ysize};
    struct cudaExtent extent = {.depth = depth,
                                .height = height,
                                .width = width};
    *result = cudaMemset3D(pptr, value, extent);
    RECORD_RESULT(integer, *result);
    return 1;
}
/* cudaMemset3DAsync ( cudaPitchedPtr pitchedDevPtr, int  value, cudaExtent extent, cudaStream_t stream = 0 ) is not implemented */
/* cudaMemsetAsync ( void* devPtr, int  value, size_t count, cudaStream_t stream = 0 ) is not implemented */
/* cudaMipmappedArrayGetSparseProperties ( cudaArraySparseProperties* sparseProperties, cudaMipmappedArray_t mipmap ) is not implemented */
/* make_cudaExtent ( size_t w, size_t h, size_t d ) should be implemented on the client side */
/* make_cudaPitchedPtr ( void* d, size_t p, size_t xsz, size_t ysz ) should be implemented on the client side */
/* make_cudaPos ( size_t x, size_t y, size_t z ) should be implemented on the client side */

/* Peer Device Memory Access */
bool_t cuda_device_can_access_peer_1_svc(int device, int peerDevice, int_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cudaDeviceCanAccessPeer");
    result->err = cudaDeviceCanAccessPeer(&result->int_result_u.data, device, peerDevice);
    return 1;
}

bool_t cuda_device_disable_peer_access_1_svc(int peerDevice, int *result, struct svc_req *rqstp)
{
    RECORD_API(int);
    RECORD_SINGLE_ARG(peerDevice);
    LOGE(LOG_DEBUG, "cudaDeviceDisablePeerAccess");
    *result = cudaDeviceDisablePeerAccess(peerDevice);
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t cuda_device_enable_peer_access_1_svc(int peerDevice, int flags, int *result, struct svc_req *rqstp)
{
    RECORD_API(cuda_device_enable_peer_access_1_argument);
    RECORD_ARG(1, peerDevice);
    RECORD_ARG(2, flags);
    LOGE(LOG_DEBUG, "cudaDeviceEnablePeerAccess");
    *result = cudaDeviceEnablePeerAccess(peerDevice, flags);
    RECORD_RESULT(integer, *result);
    return 1;
}

/* Version Management */

bool_t cuda_driver_get_version_1_svc(int_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cudaDriverGetVersion");
    result->err = cudaDriverGetVersion(&result->int_result_u.data);
    return 1;
}

bool_t cuda_runtime_get_version_1_svc(int_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cudaRuntimeGetVersion");
    result->err = cudaRuntimeGetVersion(&result->int_result_u.data);
    return 1;
}

/*extern void** __cudaRegisterFatBinary(
  void *fatCubin
);
extern void __cudaRegisterFunction(
  void **fatCubinHandle, const char *hostFun, char *deviceFun,
  const char *deviceName, int thread_limit, void *tid,
  void *bid, void *bDim, void *gDim, int *wSize
);
extern void __cudaRegisterFatBinaryEnd(
  void **fatCubinHandle
);

bool_t cuda_register_fat_binary_1_svc(rpc_fatCubin cubin, ptr_result *result, struct svc_req *rqstp)
{
    struct __fatCubin fat = {.magic = cubin.magic,
                             .seq   = cubin.seq,
                             .text  = cubin.text,
                             .data  = cubin.data,
                             .ptr   = cubin.ptr,
                             .ptr2  = cubin.ptr2,
                             .zero  = cubin.zero};
    printf("__cudaRegisterFatBinary(magic: %x, seq: %x, text: %lx, data: %lx, ptr: %lx, ptr2: %lx, zero: %lx\n",
           fat.magic, fat.seq, fat.text, fat.data, fat.ptr, fat.ptr2, fat.zero);
    //result->ptr_result_u.ptr = (uint64_t)__cudaRegisterFatBinary(&fat);
    result->err = 0;
    return 1;
}

bool_t cuda_register_function_1_svc(ptr cubinHandle, ptr hostFun, char *deviceFun, char *deviceName, int *result, struct svc_req * rqstp)
{
    LOGE(LOG_DEBUG, "__cudaRegisterFunction(fatCubinHandle=%p, hostFun=%p, devFunc=%s, deviceName=%s)\n", (void*)cubinHandle, (void*)hostFun, deviceFun, deviceName);
   // __cudaRegisterFunction((void*)cubinHandle, (void*)hostFun, deviceFun, deviceName,                            -1, NULL, NULL, NULL, NULL, NULL);
    *result = 0;
    return 1;
}

bool_t cuda_register_fat_binary_end_1_svc(ptr cubinHandle, int *result, struct svc_req * rqstp)
{
    LOGE(LOG_DEBUG, "__cudaRegisterFatBinaryEnd(fatCubinHandle=%p)\n", (void*)cubinHandle);
    //__cudaRegisterFatBinaryEnd((void*)cubinHandle);
    *result = 0;
    return 1;
}*/

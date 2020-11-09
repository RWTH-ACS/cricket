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
#ifdef WITH_IB
#include <pthread.h>
#include "cpu-ib.h"
#endif //WITH_IB

#define WITH_RECORDER

#ifdef WITH_RECORDER
#define RECORD_VOID_API \
    api_record_t *record; \
    if (list_alloc_append(&api_records, (void**)&record) != 0) { \
        LOGE(LOG_ERROR, "list allocation failed."); \
    } \
    record->function = rqstp->rq_proc; \
    record->arguments = NULL;
#define RECORD_API(ARG_TYPE) \
    api_record_t *record; \
    ARG_TYPE *arguments; \
    if (list_alloc_append(&api_records, (void**)&record) != 0) { \
        LOGE(LOG_ERROR, "list allocation failed."); \
    } \
    if ( (arguments = malloc(sizeof(ARG_TYPE))) == NULL) { \
        LOGE(LOG_ERROR, "list arguments allocation failed"); \
    } \
    record->function = rqstp->rq_proc; \
    record->arguments = arguments;
#define RECORD_RESULT(TYPE, RES) \
    record->result.TYPE = RES
#define RECORD_SINGLE_ARG(ARG) \
    *arguments = ARG
#define RECORD_ARG(NUM, ARG) \
    arguments->arg##NUM = ARG
#else
#define RECORD_API(ARG_TYPE) 
#define RECORD_RESULT(TYPE, RES)
#define RECORD_ARG(NUM, ARG)
#define RECORD_SINGLE_ARG(ARG)
#endif //WITH_RECORDER


typedef struct api_record {
    unsigned int function;
    void *arguments;
    union {
        uint64_t u64;
        void* ptr;
        int integer;
    } result;
} api_record_t;
list api_records;

typedef struct host_alloc_info {
    int cnt;
    size_t size;
    void *client_ptr;
    void *server_ptr;
} host_alloc_info_t;
static host_alloc_info_t hainfo[64];
static size_t hainfo_cnt = 1;

int server_runtime_init(void)
{
    int ret = 0;
    ret = list_init(&api_records, sizeof(api_record_t)); 
    return ret;
}

int server_runtime_deinit(void)
{
    api_record_t *record;
    printf("server api records:\n");
    for (size_t i = 0; i < api_records.length; i++) {
        record = (api_record_t*)api_records.elements[i];
        printf("api: %u ", record->function);
        switch (record->function) {
        case CUDA_MALLOC:
            printf("(cuda_malloc), arg=%zu, result=%lx", *(size_t*)record->arguments, record->result.u64);
            break;
        case CUDA_SET_DEVICE:
            printf("(cuda_set_device)");
            break;
        case CUDA_EVENT_CREATE:
            printf("(cuda_even_create)");
            break;
        case CUDA_MEMCPY_HTOD:
            printf("(cuda_memcpy_htod)");
            break;
        case CUDA_EVENT_RECORD:
            printf("(cuda_event_record)");
            break;
        case CUDA_EVENT_DESTROY:
            printf("(cuda_event_destroy)");
            break;
        case CUDA_STREAM_CREATE_WITH_FLAGS:
            printf("(cuda_stream_create_with_flags)");
            break;
        }
        printf("\n");
    }
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

#if CUDART_VERSION >= 11000
bool_t cuda_device_get_texture_lmw_1_svc(cuda_channel_format_desc fmtDesc, int device, u64_result *result, struct svc_req *rqstp)
{
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
}
#endif

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
    LOGE(LOG_DEBUG, "cudaDeviceSynchronize\n");
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
    LOGE(LOG_DEBUG, "cudaSetDevice");
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
bool_t cuda_set_valid_devices_flags_1_svc(mem_data device_arr, int len, int *result, struct svc_req *rqstp)
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
    result->str_result_u.str = malloc(256);
    LOGE(LOG_DEBUG, "cudaGetErrorString");
    str = cudaGetErrorString((cudaError_t)error);
    strncpy(result->str_result_u.str, str, 256);
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

#if CUDART_VERSION >= 11000
bool_t cuda_ctx_reset_persisting_l2cache_1_svc(int *result, struct svc_req *rqstp)
{
    RECORD_VOID_API;
    LOGE(LOG_DEBUG, "cudaCtxResetPersistingL2Cache");
    *result = cudaCtxResetPersistingL2Cache();

    RECORD_RESULT(integer, *result);
    return 1;
}
#endif

/* Requires us do call a callback on the client side to make sense */
//        int          CUDA_STREAM_ADD_CALLBACK(ptr, ptr, mem_data, int)  = 251;

/* Requires unified memory OR attaching a shared memory region. */
//        int          CUDA_STREAM_ATTACH_MEM_ASYNC(ptr, ptr, size_t, int)= 252;

/* Requires Graph API to make sense */
//        int          CUDA_STREAM_BEGIN_CAPTURE(ptr, int)                = 253;

#if CUDART_VERSION >= 11000
bool_t cuda_stream_copy_attributes_1_svc(ptr dst, ptr src, int *result, struct svc_req *rqstp)
{
    RECORD_API(cuda_stream_copy_attributes_1_argument);
    RECORD_ARG(1, dst);
    RECORD_ARG(2, src);
    LOGE(LOG_DEBUG, "cudaStreamCopyAttributes");
    *result = cudaStreamCopyAttributes(dst, src);

    RECORD_RESULT(integer, *result);
    return 1;
}
#endif

bool_t cuda_stream_create_1_svc(ptr_result *result, struct svc_req *rqstp)
{
    RECORD_VOID_API;
    LOGE(LOG_DEBUG, "cudaStreamCreate");
    result->err = cudaStreamCreate((void*)&result->ptr_result_u.ptr);
    
    RECORD_RESULT(u64, result->ptr_result_u.ptr);
    return 1;
}

bool_t cuda_stream_create_with_flags_1_svc(int flags, ptr_result *result, struct svc_req *rqstp)
{
    RECORD_API(int);
    RECORD_SINGLE_ARG(flags);
    LOGE(LOG_DEBUG, "cudaStreamCreateWithFlags");
    result->err = cudaStreamCreateWithFlags((void*)&result->ptr_result_u.ptr, flags);
    
    RECORD_RESULT(u64, result->ptr_result_u.ptr);
    return 1;
}

bool_t cuda_stream_create_with_priority_1_svc(int flags, int priority, ptr_result *result, struct svc_req *rqstp)
{
    RECORD_API(cuda_stream_create_with_priority_1_argument);
    RECORD_ARG(1, flags);
    RECORD_ARG(2, priority);

    LOGE(LOG_DEBUG, "cudaStreamCreateWithPriority");
    result->err = cudaStreamCreateWithPriority((void*)&result->ptr_result_u.ptr, flags, priority);
    
    RECORD_RESULT(u64, result->ptr_result_u.ptr);
    return 1;
}

bool_t cuda_stream_destroy_1_svc(ptr stream, int *result, struct svc_req *rqstp)
{
    RECORD_API(ptr);
    RECORD_SINGLE_ARG(stream);
    LOGE(LOG_DEBUG, "cudaStreamDestroy\n");
    *result = cudaStreamDestroy((cudaStream_t) stream);
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
    result->err = cudaStreamGetFlags((void*)hStream, (unsigned*)&result->int_result_u.data);
    return 1;
}

bool_t cuda_stream_get_priority_1_svc(ptr hStream, int_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cudaStreamGetPriority");
    result->err = cudaStreamGetPriority((void*)hStream, &result->int_result_u.data);
    return 1;
}

/* Capture API does not make sense without graph API */
//        /* ?         CUDA_STREAM_IS_CAPTURING(ptr)                      = 264;*/

bool_t cuda_stream_query_1_svc(ptr hStream, int *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cudaStreamQuery");
    *result = cudaStreamQuery((void*)hStream);
    return 1;
}

/* What datatypes are in the union cudaStreamAttrValue? */
//        /*int          CUDA_STREAM_SET_ATTRIBUTE(ptr, int, ?)             = 266;*/

bool_t cuda_stream_synchronize_1_svc(ptr stream, int *result, struct svc_req *rqstp)
{
    RECORD_API(uint64_t);
    RECORD_SINGLE_ARG(stream);
    LOGE(LOG_DEBUG, "cudaStreamSynchronize");
    *result = cudaStreamSynchronize((struct CUstream_st*)stream);
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
    *result = cudaStreamWaitEvent((void*)stream, (void*)event, flags);
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
    RECORD_RESULT(u64, result->ptr_result_u.ptr);
    return 1;
}

bool_t cuda_event_create_with_flags_1_svc(int flags, ptr_result *result, struct svc_req *rqstp)
{
    RECORD_API(int);
    RECORD_SINGLE_ARG(flags);
    LOGE(LOG_DEBUG, "cudaEventCreateWithFlags");
    result->err = cudaEventCreateWithFlags((struct CUevent_st**)&result->ptr_result_u.ptr, flags);
    RECORD_RESULT(u64, result->ptr_result_u.ptr);
    return 1;
}

bool_t cuda_event_destroy_1_svc(ptr event, int *result, struct svc_req *rqstp)
{
    RECORD_API(ptr);
    RECORD_SINGLE_ARG(event);
    LOGE(LOG_DEBUG, "cudaEventDestroy");
    *result = cudaEventDestroy((struct CUevent_st*) event);
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t cuda_event_elapsed_time_1_svc(ptr start, ptr end, float_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cudaEventElapsedTime");
    result->err = cudaEventElapsedTime(&result->float_result_u.data, (struct CUevent_st*) start, (struct CUevent_st*)end);
    return 1;
}

bool_t cuda_event_query_1_svc(ptr event, int *result, struct svc_req *rqstp)
{
    RECORD_API(ptr);
    RECORD_SINGLE_ARG(event);
    LOGE(LOG_DEBUG, "cudaEventQuery");
    *result = cudaEventQuery((struct CUevent_st*) event);
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t cuda_event_record_1_svc(ptr event, ptr stream, int *result, struct svc_req *rqstp)
{
    RECORD_API(cuda_event_record_1_argument);
    RECORD_ARG(1, event);
    RECORD_ARG(2, stream);
    LOGE(LOG_DEBUG, "cudaEventRecord");
    *result = cudaEventRecord((struct CUevent_st*) event, (struct CUstream_st*)stream);
    RECORD_RESULT(integer, *result);
    return 1;
}

#if CUDART_VERSION >= 11000
bool_t cuda_event_record_with_flags_1_svc(ptr event, ptr stream, int flags, int *result, struct svc_req *rqstp)
{
    RECORD_API(cuda_event_record_with_flags_1_argument);
    RECORD_ARG(1, event);
    RECORD_ARG(2, stream);
    RECORD_ARG(3, flags);
    LOGE(LOG_DEBUG, "cudaEventRecordWithFlags\n");
    *result = cudaEventRecordWithFlags((struct CUevent_st*) event, (struct CUstream_st*)stream, flags);
    RECORD_RESULT(integer, *result);
    return 1;
}
#endif

bool_t cuda_event_synchronize_1_svc(ptr event, int *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cudaEventSynchronize");
    *result = cudaEventSynchronize((struct CUevent_st*) event);
    return 1;
}

/**/

bool_t cuda_malloc_1_svc(size_t argp, ptr_result *result, struct svc_req *rqstp)
{
    RECORD_API(size_t);
    RECORD_SINGLE_ARG(argp);
    LOGE(LOG_DEBUG, "cudaMalloc");
    result->err = cudaMalloc((void**)&result->ptr_result_u.ptr, argp);

    RECORD_RESULT(u64, result->ptr_result_u.ptr);
    return 1;
}

bool_t cuda_free_1_svc(uint64_t ptr, int *result, struct svc_req *rqstp)
{
    int ret = 1;
    uint64_t arg;
    api_record_t *r;
    LOGE(LOG_DEBUG, "cudaFree");
    *result = cudaFree((void*)ptr);

    /* The cleanup/simplification of the record could also be
     * done during checkpoint creation. What is better depends
     * on whether we want to minimize overhead of cudaFree or
     * of the checkpoint creation. */
    for (size_t i = 0; i < api_records.length; ++i) {
        r = (api_record_t*)api_records.elements[i];
        switch (r->function) {
        case CUDA_MALLOC:
            arg = r->result.u64;
            break;
        case CUDA_MEMCPY_DTOD:
            arg = ((cuda_memcpy_dtod_1_argument*)r->arguments)->arg1;
            break;
        case CUDA_MEMCPY_HTOD:
            arg = ((cuda_memcpy_htod_1_argument*)r->arguments)->arg1;
            break;
        case CUDA_MEMCPY_TO_SYMBOL:
            arg = ((cuda_memcpy_to_symbol_1_argument*)r->arguments)->arg1;
            break;
        case CUDA_MEMCPY_TO_SYMBOL_SHM:
            arg = ((cuda_memcpy_to_symbol_shm_1_argument*)r->arguments)->arg2;
            break;
        case CUDA_MEMCPY_IB:
            arg = ((cuda_memcpy_ib_1_argument*)r->arguments)->arg2;
            break;
        case CUDA_MEMCPY_SHM:
            arg = ((cuda_memcpy_shm_1_argument*)r->arguments)->arg2;
            break;
        default:
            continue;
        }
        if (arg == ptr) {
            //list_rm(&api_records, i, NULL);
            //--i;
            ret = 0;
        }
    }
    if (ret != 0) {
        LOGE(LOG_ERROR, "could not find a malloc call associated with this free call");
        *result = CUDA_ERROR_UNKNOWN;
    }
    return 1;
}

bool_t cuda_memcpy_dtod_1_svc(ptr dst, ptr src, size_t size, int *result, struct svc_req *rqstp)
{
    RECORD_API(cuda_memcpy_dtod_1_argument);
    RECORD_ARG(1, dst);
    RECORD_ARG(2, src);
    RECORD_ARG(3, size);

    LOGE(LOG_DEBUG, "cudaMemcpyDtoD");
    *result = cudaMemcpy((void*)dst, (void*)src, size, cudaMemcpyDeviceToDevice);

    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t cuda_memcpy_htod_1_svc(uint64_t ptr, mem_data mem, size_t size, int *result, struct svc_req *rqstp)
{
    RECORD_API(cuda_memcpy_htod_1_argument);
    RECORD_ARG(1, ptr);
    RECORD_ARG(2, mem);
    RECORD_ARG(3, size);

    LOGE(LOG_DEBUG, "cudaMemcpyHtoD\n");
    if (size != mem.mem_data_len) {
        LOGE(LOG_ERROR, "data size mismatch\n");
        *result = cudaErrorUnknown;
        return 1;
    }
#ifdef WITH_MEMCPY_REGISTER
    if ((*result = cudaHostRegister(mem.mem_data_val, size, cudaHostRegisterMapped)) != cudaSuccess) {
        LOGE(LOG_ERROR, "cudaHostRegister failed: %d.", *result);
        return 1;
    }
#endif
    *result = cudaMemcpy((void*)ptr, mem.mem_data_val, size, cudaMemcpyHostToDevice);
#ifdef WITH_MEMCPY_REGISTER
    cudaHostUnregister(mem.mem_data_val);
#endif

    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t cuda_memcpy_to_symbol_1_svc(uint64_t ptr, mem_data mem, size_t size, size_t offset, int *result, struct svc_req *rqstp)
{
    RECORD_API(cuda_memcpy_to_symbol_1_argument);
    RECORD_ARG(1, ptr);
    RECORD_ARG(2, mem);
    RECORD_ARG(3, size);
    RECORD_ARG(4, offset);

    LOGE(LOG_DEBUG, "cudaMemcpyToSymbol\n");
    if (size != mem.mem_data_len) {
        LOGE(LOG_ERROR, "data size mismatch\n");
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
    LOGE(LOG_DEBUG, "cudaMemcpyToSymbolShm\n");
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

#if WITH_IB
struct ib_thread_info {
    int index;
    void* host_ptr;
    void *device_ptr;
    size_t size;
    int result;
};
void* ib_thread(void* arg)
{
    struct ib_thread_info *info = (struct ib_thread_info*)arg;
    ib_server_recv(info->host_ptr, info->index, info->size);
    info->result = cudaMemcpy(info->device_ptr, info->host_ptr, info->size, cudaMemcpyHostToDevice);
    //ib_cleanup();
    free (info);
    return NULL;
}

bool_t cuda_memcpy_ib_1_svc(int index, ptr device_ptr, size_t size, int kind, int *result, struct svc_req *rqstp)
{
    RECORD_API(cuda_memcpy_to_symbol_ib_1_argument);
    RECORD_ARG(1, index);
    RECORD_ARG(2, device_ptr);
    RECORD_ARG(3, size);
    RECORD_ARG(4, kind);
    LOGE(LOG_DEBUG, "cudaMemcpyIB\n");
    *result = cudaErrorInitializationError;
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
        struct ib_thread_info *info = malloc(sizeof(struct ib_thread_info));
        info->index = index;
        info->host_ptr = hainfo[index].server_ptr;
        info->device_ptr = (void*)device_ptr;
        info->size = size;
        info->result = 0;
        if (pthread_create(&thread, NULL, ib_thread, info) != 0) {
            LOGE(LOG_ERROR, "starting ib thread failed.");
            goto out;
        }
        *result = cudaSuccess;
    } else if (kind == cudaMemcpyDeviceToHost) {
        *result = cudaMemcpy(hainfo[index].server_ptr, (void*)device_ptr, size, kind);
        ib_client_send(hainfo[index].server_ptr, index, size, "epyc4");
    }
out:
    RECORD_RESULT(integer, *result);
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
    LOGE(LOG_DEBUG, "cudaMemcpyShm\n");
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
        *result = cudaMemcpy((void*)device_ptr, hainfo[index].server_ptr, size, kind);
    } else if (kind == cudaMemcpyDeviceToHost) {
        *result = cudaMemcpy(hainfo[index].server_ptr, (void*)device_ptr, size, kind);
    }
out:
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t cuda_memcpy_dtoh_1_svc(uint64_t ptr, size_t size, mem_result *result, struct svc_req *rqstp)
{
    //Does not need to be recorded because doesn't change device state
    LOGE(LOG_DEBUG, "cudaMemcpyDtoH(%p, %zu)\n", ptr, size);
    result->mem_result_u.data.mem_data_len = size;
    result->mem_result_u.data.mem_data_val = malloc(size);
#ifdef WITH_MEMCPY_REGISTER
    if ((result->err = cudaHostRegister(result->mem_result_u.data.mem_data_val,
                                        size, cudaHostRegisterMapped)) != cudaSuccess) {
        LOGE(LOG_ERROR, "cudaHostRegister failed.");
        goto out;
    }
#endif
    result->err = cudaMemcpy(result->mem_result_u.data.mem_data_val, (void*)ptr, size, cudaMemcpyDeviceToHost);
#ifdef WITH_MEMCPY_REGISTER
    cudaHostUnregister(result->mem_result_u.data.mem_data_val);
#endif
out: 
    return 1;
}

bool_t cuda_launch_kernel_1_svc(ptr function, rpc_dim3 gridDim, rpc_dim3 blockDim,
                                mem_data args, size_t sharedMem, ptr stream,
                                int *result, struct svc_req *rqstp)
{
    dim3 cuda_gridDim = {gridDim.x, gridDim.y, gridDim.z};
    dim3 cuda_blockDim = {blockDim.x, blockDim.y, blockDim.z};
    void **cuda_args;
    uint16_t *arg_offsets;
    size_t param_num = *((size_t*)args.mem_data_val);
    arg_offsets = (uint16_t*)(args.mem_data_val+sizeof(size_t));
    cuda_args = malloc(param_num*sizeof(void*));
    for (size_t i = 0; i < param_num; ++i) {
        cuda_args[i] = args.mem_data_val+sizeof(size_t)+param_num*sizeof(uint16_t)+arg_offsets[i];
        //LOGE(LOG_DEBUG, "arg: %p (%d)\n", *(void**)cuda_args[i], *(int*)cuda_args[i]);
    }

    LOGE(LOG_DEBUG, "cudaLaunchKernel(func=%p, gridDim=[%d,%d,%d], blockDim=[%d,%d,%d], args=%p, sharedMem=%d, stream=%p)\n", function, cuda_gridDim.x, cuda_gridDim.y, cuda_gridDim.z, cuda_blockDim.x, cuda_blockDim.y, cuda_blockDim.z, cuda_args, sharedMem, (void*)stream);

    *result = cudaLaunchKernel((void*)function, cuda_gridDim, cuda_blockDim, cuda_args, sharedMem, (void*)stream);
    LOGE(LOG_DEBUG, "cudaLaunchKernel result: %d\n", *result);
    return 1;
}



bool_t cuda_free_host_1_svc(int index, int *result, struct svc_req *rqstp)
{
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

bool_t cuda_host_alloc_1_svc(int client_cnt, size_t size, ptr client_ptr, unsigned int flags, int *result, struct svc_req *rqstp)
{
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

    if (client_cnt != hainfo_cnt) {
        LOGE(LOG_ERROR, "number of shm segments on client and server do not agree (client: %d, host: %d)", client_cnt, hainfo_cnt);
        return 1;
    }

    if (socktype == TCP) { //Use infiniband
#ifdef WITH_IB
        void *server_ptr = NULL;
        if (ib_allocate_memreg(&server_ptr, size, hainfo_cnt) == 0) {

            if (flags & cudaHostAllocPortable) {
                register_flags |= cudaHostRegisterPortable;
            }
            if (flags & cudaHostAllocMapped) {
                register_flags |= cudaHostRegisterMapped;
            }
            if (flags & cudaHostAllocWriteCombined) {
                register_flags |= cudaHostRegisterMapped;
            }

            if ((*result = cudaHostRegister(server_ptr, size, flags)) != cudaSuccess) {
                LOGE(LOG_ERROR, "cudaHostRegister failed.");
                goto out;
            }

            hainfo[hainfo_cnt].cnt = client_cnt;
            hainfo[hainfo_cnt].size = size;
            hainfo[hainfo_cnt].client_ptr = (void*)client_ptr;
            hainfo[hainfo_cnt].server_ptr = server_ptr;
            hainfo_cnt++;
        } else {
            LOGE(LOG_ERROR, "failed to register infiniband memory region");
            goto out;
        }
#else
                LOGE(LOG_ERROR, "infiniband is disabled.");
                goto cleanup;
#endif //WITH_IB

    } else if (socktype == UNIX) { //Use local shared memory
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

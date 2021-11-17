#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <driver_types.h>
#include <surface_types.h>
#include <texture_types.h>
#include <cuda_runtime_api.h>

//for strerror
#include <string.h>
#include <errno.h>

//For SHM
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "cpu-libwrap.h"
#include "cpu_rpc_prot.h"
#include "cpu-common.h"
#include "cpu-utils.h"
#include "log.h"
#ifdef WITH_IB
#include "cpu-ib.h"
#endif //WITH_IB


#ifdef WITH_API_CNT
int api_call_cnt = 0;
size_t memcpy_cnt = 0;
void cpu_runtime_print_api_call_cnt(void)
{
    LOG(LOG_INFO, "api-call-cnt: %d", api_call_cnt);
    LOG(LOG_INFO, "memcpy-cnt: %d", memcpy_cnt);
}
#endif //WITH_API_CNT


cudaError_t cudaChooseDevice(int* device, const struct cudaDeviceProp* prop)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int_result result;
    enum clnt_stat retval_1;
    mem_data prop_mem = {
      .mem_data_len = sizeof(struct cudaDeviceProp),
      .mem_data_val = (void*)prop};

    retval_1 = cuda_choose_device_1(prop_mem, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    if (result.err == 0) {
        *device = result.int_result_u.data;
    }
    return result.err;
}

cudaError_t cudaDeviceGetAttribute(int* value, enum cudaDeviceAttr attr, int device)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int_result result;
    enum clnt_stat retval_1;

    retval_1 = cuda_device_get_attribute_1((int)attr, device, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    if (result.err == 0) {
        *value = result.int_result_u.data;
    }
    return result.err;
}

cudaError_t cudaDeviceGetByPCIBusId(int* device, const char* pciBusId)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int_result result;
    enum clnt_stat retval_1;

    retval_1 = cuda_device_get_by_pci_bus_id_1((char*)pciBusId, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    if (result.err == 0) {
        *device = result.int_result_u.data;
    }
    return result.err;
}

cudaError_t cudaDeviceGetCacheConfig(enum cudaFuncCache* pCacheConfig)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int_result result;
    enum clnt_stat retval_1;

    retval_1 = cuda_device_get_cache_config_1(&result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    if (result.err == 0) {
        *pCacheConfig = result.int_result_u.data;
    }
    return result.err;
}

cudaError_t cudaDeviceGetLimit(size_t* pValue, enum cudaLimit limit)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    u64_result result;
    enum clnt_stat retval_1;

    retval_1 = cuda_device_get_limit_1(limit, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    if (result.err == 0) {
        *pValue = result.u64_result_u.u64;
    }
    return result.err;
}

cudaError_t cudaDeviceGetP2PAttribute(int* value, enum cudaDeviceP2PAttr attr, int  srcDevice, int  dstDevice)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int_result result;
    enum clnt_stat retval_1;

    retval_1 = cuda_device_get_p2p_attribute_1(attr, srcDevice, dstDevice, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    if (result.err == 0) {
        *value = result.int_result_u.data;
    }
    return result.err;
}

cudaError_t cudaDeviceGetPCIBusId(char* pciBusId, int len, int device)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    str_result result;
    enum clnt_stat retval_1;

    retval_1 = cuda_device_get_pci_bus_id_1(len, device, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    if (result.err == 0) {
        strncpy(pciBusId, result.str_result_u.str, len);
    }
    return result.err;
}

cudaError_t cudaDeviceGetSharedMemConfig(enum cudaSharedMemConfig* pConfig)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int_result result;
    enum clnt_stat retval_1;

    retval_1 = cuda_device_get_shared_mem_config_1(&result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    if (result.err == 0) {
        *pConfig = result.int_result_u.data;
    }
    return result.err;
}

cudaError_t cudaDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    dint_result result;
    enum clnt_stat retval_1;

    retval_1 = cuda_device_get_stream_priority_range_1(&result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    if (result.err == 0) {
        *leastPriority = result.dint_result_u.data.i1;
        *greatestPriority = result.dint_result_u.data.i2;
    }
    return result.err;
}

cudaError_t cudaDeviceReset(void)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval_1;
    retval_1 = cuda_device_reset_1(&result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    return result;
}

cudaError_t cudaDeviceSetCacheConfig(enum cudaFuncCache cacheConfig)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval_1;
    retval_1 = cuda_device_set_cache_config_1(cacheConfig, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    return result;
}

cudaError_t cudaDeviceSetLimit(enum cudaLimit limit, size_t value)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval_1;
    retval_1 = cuda_device_set_limit_1(limit, value, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    return result;
}

cudaError_t cudaDeviceSetSharedMemConfig(enum cudaSharedMemConfig config)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval_1;
    retval_1 = cuda_device_set_shared_mem_config_1(config, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    return result;
}

cudaError_t cudaDeviceSynchronize(void)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result = -1;
    enum clnt_stat retval_1;
    for (int i=0; result != 0 && i < 10; ++i) {
        retval_1 = cuda_device_synchronize_1(&result, clnt);
        if (retval_1 != RPC_SUCCESS) {
            clnt_perror (clnt, "call failed");
        }
    }
    return result;
}

cudaError_t cudaGetDevice(int* device)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int_result result;
    enum clnt_stat retval_1;
    retval_1 = cuda_get_device_1(&result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    if (result.err == 0) {
        *device = result.int_result_u.data;
    }
    return result.err;
}

cudaError_t cudaGetDeviceCount(int* count)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int_result result;
    enum clnt_stat retval_1;
    retval_1 = cuda_get_device_count_1(&result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    if (result.err == 0) {
        *count = result.int_result_u.data;
    }
    return result.err;
}

cudaError_t cudaGetDeviceFlags(unsigned int* flags)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int_result result;
    enum clnt_stat retval_1;
    retval_1 = cuda_get_device_flags_1(&result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    if (result.err == 0) {
        *flags = result.int_result_u.data;
    }
    return result.err;
}

cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp* prop, int device)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    mem_result result;
    result.mem_result_u.data.mem_data_len = sizeof(struct cudaDeviceProp);
    result.mem_result_u.data.mem_data_val = (char*)prop;
    enum clnt_stat retval;
    retval = cuda_get_device_properties_1(device, &result, clnt);
    if (retval != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    if (result.err != 0) {
        return result.err;
    }
    if (result.mem_result_u.data.mem_data_len != sizeof(struct cudaDeviceProp)) {
        LOGE(LOG_ERROR, "error: expected size != retrieved size\n");
        return result.err;
    }
    return result.err;
}

DEF_FN(cudaError_t, cudaIpcCloseMemHandle, void*, devPtr)
DEF_FN(cudaError_t, cudaIpcGetEventHandle, cudaIpcEventHandle_t*, handle, cudaEvent_t, event)
DEF_FN(cudaError_t, cudaIpcGetMemHandle, cudaIpcMemHandle_t*, handle, void*, devPtr)
DEF_FN(cudaError_t, cudaIpcOpenEventHandle, cudaEvent_t*, event, cudaIpcEventHandle_t, handle)
DEF_FN(cudaError_t, cudaIpcOpenMemHandle, void**, devPtr, cudaIpcMemHandle_t, handle, unsigned int,  flags)

cudaError_t cudaSetDevice(int device)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval_1;
    retval_1 = cuda_set_device_1(device, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    return result;
}

cudaError_t cudaSetDeviceFlags(unsigned int  flags)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval_1;
    retval_1 = cuda_set_device_flags_1(flags, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    return result;
}

cudaError_t cudaSetValidDevices(int *device_arr, int len)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval_1;
    mem_data arr_mem = {.mem_data_len = len*sizeof(int),
                        .mem_data_val = (void*)device_arr};
    retval_1 = cuda_set_valid_devices_1(arr_mem, len, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    return result;
}

const char* cudaGetErrorName(cudaError_t error)
{
    str_result result;
    enum clnt_stat retval_1;
    result.str_result_u.str = malloc(128);
    retval_1 = cuda_get_error_name_1(error, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    if (result.err != 0) {
        LOGE(LOG_ERROR, "something went wrong");
    }
    return result.str_result_u.str;
}

const char* cudaGetErrorString(cudaError_t error)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    str_result result;
    enum clnt_stat retval_1;
    result.str_result_u.str = malloc(256);
    retval_1 = cuda_get_error_string_1(error, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    if (result.err != 0) {
        LOGE(LOG_ERROR, "something went wrong");
    }
    return result.str_result_u.str;
}

cudaError_t cudaGetLastError(void)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval_1;
    retval_1 = cuda_get_last_error_1(&result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    return result;
}

cudaError_t cudaPeekAtLastError(void)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval_1;
    retval_1 = cuda_peek_at_last_error_1(&result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    return result;
}

DEF_FN(cudaError_t, cudaStreamAddCallback, cudaStream_t, stream, cudaStreamCallback_t, callback, void*, userData, unsigned int,  flags)
DEF_FN(cudaError_t, cudaStreamAttachMemAsync, cudaStream_t, stream, void*, devPtr, size_t, length, unsigned int,  flags)
DEF_FN(cudaError_t, cudaStreamBeginCapture, cudaStream_t, stream, enum cudaStreamCaptureMode, mode)

cudaError_t cudaStreamCreate(cudaStream_t* pStream)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    ptr_result result;
    enum clnt_stat retval_1;
    retval_1 = cuda_stream_create_1(&result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    if (result.err == 0) {
        *pStream = (void*)result.ptr_result_u.ptr;
    }
    return result.err;
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int flags)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    ptr_result result;
    enum clnt_stat retval_1;
    retval_1 = cuda_stream_create_with_flags_1(flags, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    if (result.err == 0) {
        *pStream = (void*)result.ptr_result_u.ptr;
    }
    return result.err;
}

cudaError_t cudaStreamCreateWithPriority(cudaStream_t* pStream, unsigned int  flags, int  priority)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    ptr_result result;
    enum clnt_stat retval_1;
    retval_1 = cuda_stream_create_with_priority_1(flags, priority, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    if (result.err == 0) {
        *pStream = (void*)result.ptr_result_u.ptr;
    }
    return result.err;
}

cudaError_t cudaStreamDestroy(cudaStream_t stream)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval_1;
    retval_1 = cuda_stream_destroy_1((ptr)stream, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    return result;
}

DEF_FN(cudaError_t, cudaStreamEndCapture, cudaStream_t, stream, cudaGraph_t*, pGraph)
#if CUDART_VERSION >= 11000
DEF_FN(cudaError_t, cudaStreamGetAttribute, cudaStream_t, hStream, enum cudaStreamAttrID, attr, union cudaStreamAttrValue*, value_out )
#endif
DEF_FN(cudaError_t, cudaStreamGetCaptureInfo, cudaStream_t, stream, cudaStreamCaptureStatus**, pCaptureStatus, unsigned long, long*, pId)

cudaError_t cudaStreamGetFlags(cudaStream_t hStream, unsigned int* flags)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int_result result;
    enum clnt_stat retval_1;
    retval_1 = cuda_stream_get_flags_1((ptr)hStream, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    if (result.err == 0) {
        *flags = (unsigned)result.int_result_u.data;
    }
    return result.err;
}

cudaError_t cudaStreamGetPriority(cudaStream_t hStream, int* priority)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int_result result;
    enum clnt_stat retval_1;
    retval_1 = cuda_stream_get_priority_1((ptr)hStream, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    if (result.err == 0) {
        *priority = result.int_result_u.data;
    }
    return result.err;
}

DEF_FN(cudaError_t, cudaStreamIsCapturing, cudaStream_t, stream, enum cudaStreamCaptureStatus*, pCaptureStatus)

cudaError_t cudaStreamQuery(cudaStream_t stream)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval_1;
    retval_1 = cuda_stream_query_1((ptr)stream, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    return result;
}

#if CUDART_VERSION >= 11000
DEF_FN(cudaError_t, cudaStreamSetAttribute, cudaStream_t, hStream, enum cudaStreamAttrID, attr, const union cudaStreamAttrValue*, value)
#endif

cudaError_t cudaStreamSynchronize(cudaStream_t stream)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval_1;
    retval_1 = cuda_stream_synchronize_1((ptr)stream, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    return result;
}

cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval_1;
    retval_1 = cuda_stream_wait_event_1((ptr)stream, (ptr)event, flags, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    return result;
}

cudaError_t cudaThreadExchangeStreamCaptureMode(enum cudaStreamCaptureMode* mode)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int_result result;
    enum clnt_stat retval_1;
    retval_1 = cuda_thread_exchange_stream_capture_mode_1(*mode, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    if (result.err == 0) {
        *mode = result.int_result_u.data;
    }
    return result.err;
}

cudaError_t cudaEventCreate(cudaEvent_t* event)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    ptr_result result;
    enum clnt_stat retval_1;
    retval_1 = cuda_event_create_1(&result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    if (result.err == 0) {
        *event = (void*)result.ptr_result_u.ptr;
    }
    return result.err;
}

cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int  flags)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    ptr_result result;
    enum clnt_stat retval_1;
    retval_1 = cuda_event_create_with_flags_1(flags, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    if (result.err == 0) {
        *event = (void*)result.ptr_result_u.ptr;
    }
    return result.err;
}

cudaError_t cudaEventDestroy(cudaEvent_t event)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval_1;
    retval_1 = cuda_event_destroy_1((ptr)event, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    return result;
}

cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    float_result result;
    enum clnt_stat retval_1;
    retval_1 = cuda_event_elapsed_time_1((ptr)start, (ptr)end, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    if (result.err == 0) {
        *ms = result.float_result_u.data;
    }
    return result.err;
}

cudaError_t cudaEventQuery(cudaEvent_t event)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval_1;
    retval_1 = cuda_event_query_1((ptr)event, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    return result;
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval_1;
    retval_1 = cuda_event_record_1((ptr)event, (ptr)stream, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    return result;
}

cudaError_t cudaEventSynchronize(cudaEvent_t event)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result = -1;
    enum clnt_stat retval_1;
    for (int i=0; result < 0 && i < 10; ++i) {
        retval_1 = cuda_event_synchronize_1((ptr)event, &result, clnt);
        if (retval_1 != RPC_SUCCESS) {
            clnt_perror (clnt, "call failed");
        }
    }
    return result;
}

DEF_FN(cudaError_t, cudaDestroyExternalMemory, cudaExternalMemory_t, extMem)
DEF_FN(cudaError_t, cudaDestroyExternalSemaphore, cudaExternalSemaphore_t, extSem)
DEF_FN(cudaError_t, cudaExternalMemoryGetMappedBuffer, void**, devPtr, cudaExternalMemory_t, extMem, const struct cudaExternalMemoryBufferDesc*, bufferDesc)
DEF_FN(cudaError_t, cudaExternalMemoryGetMappedMipmappedArray, cudaMipmappedArray_t*, mipmap, cudaExternalMemory_t, extMem, const struct cudaExternalMemoryMipmappedArrayDesc*, mipmapDesc)
DEF_FN(cudaError_t, cudaImportExternalMemory, cudaExternalMemory_t*, extMem_out, const struct cudaExternalMemoryHandleDesc*, memHandleDesc)
DEF_FN(cudaError_t, cudaImportExternalSemaphore, cudaExternalSemaphore_t*, extSem_out, const struct cudaExternalSemaphoreHandleDesc*, semHandleDesc)
DEF_FN(cudaError_t, cudaSignalExternalSemaphoresAsync, const cudaExternalSemaphore_t*, extSemArray, const struct cudaExternalSemaphoreSignalParams*, paramsArray, unsigned int,  numExtSems, cudaStream_t, stream)
DEF_FN(cudaError_t, cudaWaitExternalSemaphoresAsync, const cudaExternalSemaphore_t*, extSemArray, const struct cudaExternalSemaphoreWaitParams*, paramsArray, unsigned int,  numExtSems, cudaStream_t, stream)

cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes* attr, const void* func)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    mem_result result;
    enum clnt_stat retval_1;
    retval_1 = cuda_func_get_attributes_1((ptr)func, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    if (result.err == 0) {
        memcpy(attr, result.mem_result_u.data.mem_data_val,
                     result.mem_result_u.data.mem_data_len);
    }
    return result.err;
}

cudaError_t cudaFuncSetAttribute(const void* func, enum cudaFuncAttribute attr, int value)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval_1;
    retval_1 = cuda_func_set_attributes_1((ptr)func, attr, value, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    return result;
}

cudaError_t cudaFuncSetCacheConfig(const void* func, enum cudaFuncCache cacheConfig)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval_1;
    retval_1 = cuda_func_set_cache_config_1((ptr)func, cacheConfig, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    return result;
}

cudaError_t cudaFuncSetSharedMemConfig(const void* func, enum cudaSharedMemConfig config)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval_1;
    retval_1 = cuda_func_set_shared_mem_config_1((ptr)func, config, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    return result;
}

cudaError_t cudaLaunchCooperativeKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval_1;
    size_t i;
    char *buf;
    kernel_info_t *info;
    int found_kernel = 0;

    for (i=0; i < kernel_infos.length; ++i) {
        if (list_at(&kernel_infos, i, (void**)&info) != 0) {
            LOGE(LOG_ERROR, "error gettint element at %d", i);
            return cudaErrorInvalidDeviceFunction;
        }
        if (func != NULL && info != NULL && info->host_fun == func) {
            LOG(LOG_DEBUG, "calling kernel \"%s\" (param_size: %zd, param_num: %zd)", info->name, info->param_size, info->param_num);
            found_kernel = 1;
            break;
        }
    }
    if (!found_kernel) {
        LOGE(LOG_ERROR, "request to call unknown kernel.");
        return cudaErrorInvalidDeviceFunction;
    }

    rpc_dim3 rpc_gridDim = {gridDim.x, gridDim.y, gridDim.z};
    rpc_dim3 rpc_blockDim = {blockDim.x, blockDim.y, blockDim.z};
    mem_data rpc_args;
    rpc_args.mem_data_len = sizeof(size_t)+info->param_num*sizeof(uint16_t)+info->param_size;
    rpc_args.mem_data_val = malloc(rpc_args.mem_data_len);
    memcpy(rpc_args.mem_data_val, &info->param_num, sizeof(size_t));
    memcpy(rpc_args.mem_data_val + sizeof(size_t), info->param_offsets, info->param_num*sizeof(uint16_t));
    for (size_t j=0, size=0; j < info->param_num; ++j) {
        size = info->param_sizes[j];
        memcpy(rpc_args.mem_data_val + sizeof(size_t) + info->param_num*sizeof(uint16_t) +
               info->param_offsets[j],
               args[j],
               size);
    }
    retval_1 = cuda_launch_cooperative_kernel_1((uint64_t)func, rpc_gridDim, rpc_blockDim, rpc_args, sharedMem, (uint64_t)stream, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    free(rpc_args.mem_data_val);
    return result;
}

cudaError_t cudaLaunchCooperativeKernelMultiDevice(struct cudaLaunchParams* launchParamsList, unsigned int numDevices, unsigned int flags)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval_1;
    size_t i;
    char *buf;
    void* func = launchParamsList->func;
    kernel_info_t *info;
    int found_kernel = 0;
    dim3 gridDim = launchParamsList->gridDim;
    dim3 blockDim = launchParamsList->blockDim;

    for (i=0; i < kernel_infos.length; ++i) {
        if (list_at(&kernel_infos, i, (void**)&info) != 0) {
            LOGE(LOG_ERROR, "error gettint element at %d", i);
            return cudaErrorInvalidDeviceFunction;
        }
        if (func != NULL && info != NULL && info->host_fun == func) {
            LOG(LOG_DEBUG, "calling kernel \"%s\" (param_size: %zd, param_num: %zd)", info->name, info->param_size, info->param_num);
            found_kernel = 1;
            break;
        }
    }
    if (!found_kernel) {
        LOGE(LOG_ERROR, "request to call unknown kernel.");
        return cudaErrorInvalidDeviceFunction;
    }

    rpc_dim3 rpc_gridDim = {gridDim.x, gridDim.y, gridDim.z};
    rpc_dim3 rpc_blockDim = {blockDim.x, blockDim.y, blockDim.z};
    mem_data rpc_args;
    rpc_args.mem_data_len = sizeof(size_t)+info->param_num*sizeof(uint16_t)+info->param_size;
    rpc_args.mem_data_val = malloc(rpc_args.mem_data_len);
    memcpy(rpc_args.mem_data_val, &info->param_num, sizeof(size_t));
    memcpy(rpc_args.mem_data_val + sizeof(size_t), info->param_offsets, info->param_num*sizeof(uint16_t));
    for (size_t j=0, size=0; j < info->param_num; ++j) {
        size = info->param_sizes[j];
        memcpy(rpc_args.mem_data_val + sizeof(size_t) + info->param_num*sizeof(uint16_t) +
               info->param_offsets[j],
               launchParamsList->args[j],
               size);
    }
    retval_1 = cuda_launch_cooperative_kernel_1(
      (uint64_t)func, rpc_gridDim, rpc_blockDim,
      rpc_args, launchParamsList->sharedMem,
      (uint64_t)launchParamsList->stream, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    free(rpc_args.mem_data_val);
    return result;
}

DEF_FN(cudaError_t, cudaLaunchHostFunc, cudaStream_t, stream, cudaHostFn_t, fn, void*, userData)

cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval_1;
    size_t i;
    char *buf;
    int found_kernel = 0;
    kernel_info_t *info;
    //printf("cudaLaunchKernel(func=%p, gridDim=[%d,%d,%d], blockDim=[%d,%d,%d], args=%p->%p,%p, sharedMem=%d, stream=%p)\n", func, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, args, args[0], args[1], sharedMem, stream);
    LOGE(LOG_DEBUG, "cudaLaunchKernel(%p)", func);

    for (i=0; i < kernel_infos.length; ++i) {
        if (list_at(&kernel_infos, i, (void**)&info) != 0) {
            LOGE(LOG_ERROR, "error gettint element at %d", i);
            return cudaErrorInvalidDeviceFunction;
        }
        if (func != NULL && info != NULL && info->host_fun == func) {
            LOG(LOG_DEBUG, "calling kernel \"%s\" (param_size: %zd, param_num: %zd)", info->name, info->param_size, info->param_num);
            found_kernel = 1;
            break;
        }
    }

    if (!found_kernel) {
        LOGE(LOG_ERROR, "request to call unknown kernel.");
        return cudaErrorInvalidDeviceFunction;
    }

    rpc_dim3 rpc_gridDim = {gridDim.x, gridDim.y, gridDim.z};
    rpc_dim3 rpc_blockDim = {blockDim.x, blockDim.y, blockDim.z};
    mem_data rpc_args;
    rpc_args.mem_data_len = sizeof(size_t)+info->param_num*sizeof(uint16_t)+info->param_size;
    rpc_args.mem_data_val = malloc(rpc_args.mem_data_len);
    memcpy(rpc_args.mem_data_val, &info->param_num, sizeof(size_t));
    memcpy(rpc_args.mem_data_val + sizeof(size_t), info->param_offsets, info->param_num*sizeof(uint16_t));
    for (size_t j=0, size=0; j < info->param_num; ++j) {
        size = info->param_sizes[j];
        //printf("p%d - size: %d, offset: %d\n", j, size, infos[i].param_offsets[j]);
        memcpy(rpc_args.mem_data_val + sizeof(size_t) + info->param_num*sizeof(uint16_t) +
               info->param_offsets[j],
               args[j],
               size);
    }
    retval_1 = cuda_launch_kernel_1((uint64_t)func, rpc_gridDim, rpc_blockDim, rpc_args, sharedMem, (uint64_t)stream, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    free(rpc_args.mem_data_val);
    return result;
}

DEF_FN(cudaError_t, cudaSetDoubleForDevice, double*, d)
DEF_FN(cudaError_t, cudaSetDoubleForHost, double*, d)
    

cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, const void* func, int  blockSize, size_t dynamicSMemSize)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int_result result;
    enum clnt_stat retval_1;
    retval_1 = cuda_occupancy_max_active_bpm_1((ptr)func, blockSize, dynamicSMemSize, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    if (result.err == 0) {
        *numBlocks = result.int_result_u.data;
    }
    return result.err;
}

cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, const void* func, int  blockSize, size_t dynamicSMemSize, unsigned int  flags)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int_result result;
    enum clnt_stat retval_1;
    retval_1 = cuda_occupancy_max_active_bpm_with_flags_1((ptr)func, blockSize, dynamicSMemSize, flags, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    if (result.err == 0) {
        *numBlocks = result.int_result_u.data;
    }
    return result.err;
}

cudaError_t cudaArrayGetInfo(struct cudaChannelFormatDesc* desc, struct cudaExtent* extent, unsigned int* flags, cudaArray_t array)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    mem_result result;
    enum clnt_stat retval_1;
    retval_1 = cuda_array_get_info_1((ptr)array, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    if (result.err == 0) {
        memcpy(desc, result.mem_result_u.data.mem_data_val, 
               sizeof(struct cudaChannelFormatDesc));
        memcpy(extent, result.mem_result_u.data.mem_data_val +
               sizeof(struct cudaChannelFormatDesc),
               sizeof(struct cudaExtent));
        memcpy(flags, result.mem_result_u.data.mem_data_val +
               sizeof(struct cudaChannelFormatDesc),
               sizeof(struct cudaExtent));
    }
    return result.err;
}

#if CUDART_VERSION >= 11000
cudaError_t cudaArrayGetSparseProperties(struct cudaArraySparseProperties* sparseProperties, cudaArray_t array)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    mem_result result;
    enum clnt_stat retval_1;
    retval_1 = cuda_array_get_info_1((ptr)array, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    if (result.err == 0) {
        memcpy(sparseProperties, result.mem_result_u.data.mem_data_val, 
               sizeof(struct cudaArraySparseProperties));
    }
    return result.err;
}
#endif

cudaError_t cudaFree(void *devPtr)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval_1;
    retval_1 = cuda_free_1((uint64_t)devPtr, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    return result;
}

cudaError_t cudaFreeArray(cudaArray_t array)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval_1;
    retval_1 = cuda_free_array_1((uint64_t)array, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    return result;
}

typedef struct host_alloc_info {
    int cnt;
    size_t size;
    void *client_ptr;
} host_alloc_info_t;
static host_alloc_info_t hainfo[64] = {0};
static size_t hainfo_cnt = 1;
static int hainfo_getindex(void *client_ptr)
{
    int i;
    for (i=0; i < hainfo_cnt; ++i) {
        if (hainfo[i].client_ptr != 0 &&
            hainfo[i].client_ptr == client_ptr) {

            return i;
        }
    }
    return -1;
}
cudaError_t cudaFreeHost(void* ptr)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result = cudaErrorInitializationError;
    enum clnt_stat retval_1;
    int i = -1;
    if (shm_enabled && connection_is_local == 1) { //Use local shared memory
        i = hainfo_getindex((void*)ptr);
        if (i == -1) {
            goto out;
        }
        munmap(hainfo[i].client_ptr, hainfo[i].size);
        memset(&hainfo[i], 0, sizeof(host_alloc_info_t));

        retval_1 = cuda_free_host_1(i, &result, clnt);
        if (retval_1 != RPC_SUCCESS) {
            clnt_perror (clnt, "call failed");
        }
        if (result != cudaSuccess) {
            LOGE(LOG_ERROR, "cudaFreeHost failed on server-side.");
            goto out;
        }
    } else if (socktype == TCP) {
#ifdef WITH_IB //Use infiniband
        i = hainfo_getindex((void*)ptr);
        if (i == -1) {
            goto out;
        }
        ib_free_memreg((void*)ptr, i, false);
        hainfo[i].client_ptr = 0;
        return cudaSuccess;
#else
        free(ptr);
        return cudaSuccess;
#endif //WITH_IB
    } else {
        free(ptr);
        return cudaSuccess;
    }
    result = cudaSuccess;
 out:
//    return result;
      return cudaSuccess;
}
DEF_FN(cudaError_t, cudaFreeMipmappedArray, cudaMipmappedArray_t, mipmappedArray)
DEF_FN(cudaError_t, cudaGetMipmappedArrayLevel, cudaArray_t*, levelArray, cudaMipmappedArray_const_t, mipmappedArray, unsigned int,  level)

cudaError_t cudaGetSymbolAddress(void** devPtr, const void* symbol)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    ptr_result result;
    enum clnt_stat retval_1;
    retval_1 = cuda_get_symbol_address_1((ptr)symbol, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    if (result.err == 0) {
        *devPtr = (void*)result.ptr_result_u.ptr;
    }
    return result.err;
}

cudaError_t cudaGetSymbolSize(size_t* size, const void* symbol)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    u64_result result;
    enum clnt_stat retval_1;
    retval_1 = cuda_get_symbol_size_1((ptr)symbol, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    if (result.err == 0) {
        *size = result.u64_result_u.u64;
    }
    return result.err;
}


cudaError_t cudaHostAlloc(void** pHost, size_t size, unsigned int flags)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int ret = cudaErrorMemoryAllocation;
    int fd_shm;
    char shm_name[128];
    enum clnt_stat retval_1;
    
    if (shm_enabled && connection_is_local == 1) { //Use local shared memory

        snprintf(shm_name, 128, "/crickethostalloc-%zu", hainfo_cnt);
        if ((fd_shm = shm_open(shm_name, O_RDWR | O_CREAT, S_IRWXU)) == -1) {
            LOGE(LOG_ERROR, "ERROR: could not open shared memory \"%s\" with size %d: %s", shm_name, size, strerror(errno));
            goto out;
        }
        if (ftruncate(fd_shm, size) == -1) {
            LOGE(LOG_ERROR, "ERROR: cannot resize shared memory");
            shm_unlink(shm_name);
            goto out;
        }
        LOGE(LOG_DEBUG, "shm opened with name \"%s\", size: %d", shm_name, size);
        if ((*pHost = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_shm, 0)) == MAP_FAILED) {
            LOGE(LOG_ERROR, "ERROR: mmap returned unexpected pointer: %p", *pHost);
            shm_unlink(shm_name);
            goto out;
        }

        hainfo[hainfo_cnt].cnt = hainfo_cnt;
        hainfo[hainfo_cnt].size = size;
        hainfo[hainfo_cnt].client_ptr = *pHost;

        retval_1 = cuda_host_alloc_1(hainfo_cnt, size, (uint64_t)*pHost, flags, &ret, clnt);
        if (retval_1 != RPC_SUCCESS) {
            clnt_perror (clnt, "call failed");
        }
        if (ret == cudaSuccess) {
            hainfo_cnt++;
        } else {
            munmap(*pHost, size);
            *pHost = NULL;
        }
        shm_unlink(shm_name);
    } else if (socktype == TCP) { //Use infiniband
#ifdef WITH_IB
       if (ib_allocate_memreg(pHost, size, hainfo_cnt, false) != 0) {
            LOGE(LOG_ERROR, "failed to register infiniband memory region");
            goto out;
        }
        hainfo[hainfo_cnt].cnt = hainfo_cnt;
        hainfo[hainfo_cnt].size = size;
        hainfo[hainfo_cnt].client_ptr = *pHost;

        hainfo_cnt++;

        retval_1 = RPC_SUCCESS;
        ret = cudaSuccess;

#else
        LOGE(LOG_DEBUG, "cudaHostAlloc is not supported for TCP transports without IB. Using malloc instead...");
        *pHost = malloc(size);
        if (*pHost == NULL) {
            goto out;
        } else {
            ret = cudaSuccess;
            goto out;
        }
#endif //WITH_IB
    } else {
        LOGE(LOG_ERROR, "unknown transport.");
        goto out;
    }
out:
    return ret;
}

cudaError_t cudaHostGetDevicePointer(void** pDevice, void* pHost, unsigned int flags)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    ptr_result result;
    enum clnt_stat retval_1;
    retval_1 = cuda_host_get_device_pointer_1((ptr)pHost, flags, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    if (result.err == 0) {
        *pDevice = (void*)result.ptr_result_u.ptr;
    }
    return result.err;
}

cudaError_t cudaHostGetFlags(unsigned int* pFlags, void* pHost)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int_result result;
    enum clnt_stat retval_1;
    retval_1 = cuda_host_get_flags_1((ptr)pHost, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    if (result.err == 0) {
        *pFlags = result.int_result_u.data;
    }
    return result.err;
}

DEF_FN(cudaError_t, cudaHostRegister, void*, ptr, size_t, size, unsigned int,  flags)
DEF_FN(cudaError_t, cudaHostUnregister, void*, ptr)


cudaError_t cudaMalloc(void** devPtr, size_t size)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    ptr_result result;
    enum clnt_stat retval_1;
    retval_1 = cuda_malloc_1(size, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    if (result.err != 0) {
        return result.err;
    }
    *devPtr = (void*)result.ptr_result_u.ptr;
    return result.err;
}

cudaError_t cudaMalloc3D(struct cudaPitchedPtr* pitchedDevPtr, struct cudaExtent extent)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    pptr_result result;
    enum clnt_stat retval_1;
    retval_1 = cuda_malloc_3d_1(extent.depth, extent.height, extent.width, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    if (result.err == 0) {
        pitchedDevPtr->pitch = result.pptr_result_u.ptr.pitch;
        pitchedDevPtr->ptr = (void*)result.pptr_result_u.ptr.ptr;
        pitchedDevPtr->xsize = result.pptr_result_u.ptr.xsize;
        pitchedDevPtr->ysize = result.pptr_result_u.ptr.pitch;
    }
    return result.err;
}

cudaError_t cudaMalloc3DArray(cudaArray_t* array, const struct cudaChannelFormatDesc* desc, struct cudaExtent extent, unsigned int flags)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    ptr_result result;
    enum clnt_stat retval_1;
    cuda_channel_format_desc rpc_desc = {
        .f = desc->f,
        .w = desc->w,
        .x = desc->x,
        .y = desc->y,
        .z = desc->z};
    retval_1 = cuda_malloc_3d_array_1(rpc_desc, extent.depth, extent.height, extent.width, flags, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    if (result.err == 0) {
        *array = (void*)result.ptr_result_u.ptr;
    }
    return result.err;
}

cudaError_t cudaMallocArray(cudaArray_t* array, const struct cudaChannelFormatDesc* desc, size_t width, size_t height, unsigned int flags)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    ptr_result result;
    enum clnt_stat retval_1;
    cuda_channel_format_desc rpc_desc = {
        .f = desc->f,
        .w = desc->w,
        .x = desc->x,
        .y = desc->y,
        .z = desc->z};
    retval_1 = cuda_malloc_array_1(rpc_desc, width, height, flags, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    if (result.err == 0) {
        *array = (void*)result.ptr_result_u.ptr;
    }
    return result.err;
}

DEF_FN(cudaError_t, cudaMallocHost, void**, ptr, size_t, size)
DEF_FN(cudaError_t, cudaMallocManaged, void**, devPtr, size_t, size, unsigned int,  flags)
DEF_FN(cudaError_t, cudaMallocMipmappedArray, cudaMipmappedArray_t*, mipmappedArray, const struct cudaChannelFormatDesc*, desc, struct cudaExtent, extent, unsigned int,  numLevels, unsigned int,  flags)

cudaError_t cudaMallocPitch(void** devPtr, size_t* pitch, size_t width, size_t height)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    ptrsz_result result;
    enum clnt_stat retval_1;
    retval_1 = cuda_malloc_pitch_1(width, height, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    if (result.err == 0) {
        *devPtr = (void*)result.ptrsz_result_u.data.p;
        *pitch = result.ptrsz_result_u.data.s;
    }
    return result.err;
}

cudaError_t cudaMemAdvise(const void* devPtr, size_t count, enum cudaMemoryAdvise advice, int device)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval_1;
    retval_1 = cuda_mem_advise_1((ptr)devPtr, count, advice, device, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    return result;
}

cudaError_t cudaMemGetInfo(size_t* free, size_t* total)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    dsz_result result;
    enum clnt_stat retval_1;
    retval_1 = cuda_mem_get_info_1(&result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    if (result.err == 0) {
        *free = result.dsz_result_u.data.sz1;
        *total = result.dsz_result_u.data.sz2;
    }
    return result.err;
}

cudaError_t cudaMemPrefetchAsync(const void* devPtr, size_t count, int dstDevice, cudaStream_t stream)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval_1;
    retval_1 = cuda_mem_prefetch_async_1((ptr)devPtr, count, dstDevice, (ptr)stream, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    return result;
}

DEF_FN(cudaError_t, cudaMemRangeGetAttribute, void*, data, size_t, dataSize, enum cudaMemRangeAttribute, attribute, const void*, devPtr, size_t, count)
DEF_FN(cudaError_t, cudaMemRangeGetAttributes, void**, data, size_t*, dataSizes, enum cudaMemRangeAttribute*, attributes, size_t, numAttributes, const void*, devPtr, size_t, count)
/* NOT IMPLEMENTED because requires unified memory
 * cudaError_t cudaMemRangeGetAttribute(void* data, size_t dataSize, enum cudaMemRangeAttribute attribute, const void* devPtr, size_t count)
 * {
 * #ifdef WITH_API_CNT
 *     api_call_cnt++;
 * #endif //WITH_API_CNT
 *     mem_result result;
 *     enum clnt_stat retval_1;
 *     retval_1 = cuda_mem_range_get_attribute_1(attribute, (ptr)devPtr, count, &result, clnt);
 *     if (retval_1 != RPC_SUCCESS) {
 *         clnt_perror (clnt, "call failed");
 *     }
 *     if (result.err == 0) {
 *         if (dataSize < result.mem_result_u.data.mem_data_len) {
 *             LOGE(LOG_ERROR, "data is too small to store result");
 *             return cudaErrorUnknown;
 *         }
 *         memcpy(data, result.mem_result_u.data.mem_data_val,
 *                      result.mem_result_u.data.mem_data_len);
 *     }
 *     return result.err;
 * }
 * cudaError_t cudaMemRangeGetAttributes(void** data, size_t* dataSizes, enum cudaMemRangeAttribute* attributes, size_t numAttributes, const void* devPtr, size_t count)
 * {
 * #ifdef WITH_API_CNT
 *     api_call_cnt++;
 * #endif //WITH_API_CNT
 *     mem_result result;
 *     enum clnt_stat retval_1;
 *     for (size_t i=0; i < numAttributes; ++i) {
 *         retval_1 = cuda_mem_range_get_attribute_1(attributes[i], (ptr)devPtr, count, &result, clnt);
 *         if (retval_1 != RPC_SUCCESS) {
 *             clnt_perror (clnt, "call failed");
 *         }
 *         if (result.err == 0) {
 *             if (dataSizes[i] < result.mem_result_u.data.mem_data_len) {
 *                 LOGE(LOG_ERROR, "data is too small to store result");
 *                 return cudaErrorUnknown;
 *             }
 *             memcpy(data, result.mem_result_u.data.mem_data_val,
 *                          result.mem_result_u.data.mem_data_len);
 *         } else {
 *             return result.err;
 *         }
 *     }
 *     return result.err;
 * }
 **/

#ifdef WITH_IB
struct ib_thread_info {
    int index;
    void* host_ptr;
    size_t size;
};
void* ib_thread(void* arg)
{
    struct ib_thread_info *info = (struct ib_thread_info*)arg;
    ib_responder_recv(info->host_ptr, info->index, info->size, false);
    //ib_cleanup();
    return NULL;
}
#endif //WITH_IB

cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind)
{
    
#ifdef WITH_API_CNT
    api_call_cnt++;
    memcpy_cnt += count;
#endif //WITH_API_CNT
    int ret = 1;
    enum clnt_stat retval;
    if (kind == cudaMemcpyHostToDevice) {
//get index of mem reg (src: cpu reg memregion)
        int index = hainfo_getindex((void*)src);
//         not a cudaHostAlloc'ed memory 
        if (index == -1) {
            mem_data src_mem;
            src_mem.mem_data_len = count;
            src_mem.mem_data_val = (void*)src;
            retval = cuda_memcpy_htod_1((uint64_t)dst, src_mem, count, &ret, clnt);
        } else {
            if (shm_enabled && connection_is_local == 1) { //Use local shared memory
                retval = cuda_memcpy_shm_1(index, (ptr)dst, count, kind, &ret, clnt);
            } else if (socktype == TCP) { //Use infiniband
#ifdef WITH_IB
                //the following commend connects to serverside cuda_memcpy_ib_1_svc, server thread is initialized waiting for client send
                retval = cuda_memcpy_ib_1(index, (ptr)dst, count, kind, &ret, clnt);
                ib_requester_send((void*)src, index, count, false);
                //ib_cleanup();
#else
                LOGE(LOG_ERROR, "infiniband is disabled.");
                goto cleanup;
#endif //WITH_IB
            }
        }
        if (retval != RPC_SUCCESS) {
            LOGE(LOG_ERROR, "RPC failed.");
            clnt_perror (clnt, "call failed");
        }
    } else if (kind == cudaMemcpyDeviceToHost) {
        //get the dst, reg cpu mem reg/ buffer
        int index = hainfo_getindex(dst);
        /* not a cudaHostAlloc'ed memory */
        if (index == -1) {

            mem_result result;
            result.mem_result_u.data.mem_data_len = count;
            result.mem_result_u.data.mem_data_val = dst;
            //printf("cuda_memcpy_dtoh(%p, %zu)\n", src, count);
            retval = cuda_memcpy_dtoh_1((uint64_t)src, count, &result, clnt);
            ret = result.err;
            if (result.err != 0) {
                goto cleanup;
            }
            if (result.mem_result_u.data.mem_data_len != count) {
                LOGE(LOG_ERROR, "error");
                goto cleanup;
            }
        } else {
            if (shm_enabled && connection_is_local) { //Use local shared memory
                retval = cuda_memcpy_shm_1(index, (ptr)src, count, kind, &ret, clnt);
            } else if (socktype == TCP) { //Use infiniband
#ifdef WITH_IB
                pthread_t thread = {0};
                struct ib_thread_info info = {
                    .index = index,
                    .host_ptr = dst,
                    .size = count,
                };
                // initialize server waiting for dev to host transfer
               if (pthread_create(&thread, NULL, ib_thread, &info) != 0) {
                    LOGE(LOG_ERROR, "starting ib thread failed.");
                    goto cleanup;
                }

                //this lead to server side function cuda_memcpy_ib_1 and sends server data to initialiced waiting ib server
                retval = cuda_memcpy_ib_1(index, (ptr)src, count, kind, &ret, clnt);
                pthread_join(thread, NULL);
#else
                LOGE(LOG_ERROR, "infiniband is disabled.");
                goto cleanup;
#endif //WITH_IB
            }

        }
        if (retval != RPC_SUCCESS) {
            LOGE(LOG_ERROR, "RPC failed.");
            clnt_perror (clnt, "call failed");
        }
    } else if (kind == cudaMemcpyDeviceToDevice) {
        retval = cuda_memcpy_dtod_1((ptr)dst, (ptr)src, count, &ret, clnt);
        if (retval != RPC_SUCCESS) {
            LOGE(LOG_ERROR, "RPC failed.");
            clnt_perror (clnt, "call failed");
        }
    } else {
        LOGE(LOG_ERROR, "unknown kind");
    }
cleanup:
    return ret;
}
DEF_FN(cudaError_t, cudaMemcpy2D, void*, dst, size_t, dpitch, const void*, src, size_t, spitch, size_t, width, size_t, height, enum cudaMemcpyKind, kind)
DEF_FN(cudaError_t, cudaMemcpy2DArrayToArray, cudaArray_t, dst, size_t, wOffsetDst, size_t, hOffsetDst, cudaArray_const_t, src, size_t, wOffsetSrc, size_t, hOffsetSrc, size_t, width, size_t, height, enum cudaMemcpyKind, kind)
DEF_FN(cudaError_t, cudaMemcpy2DAsync, void*, dst, size_t, dpitch, const void*, src, size_t, spitch, size_t, width, size_t, height, enum cudaMemcpyKind, kind, cudaStream_t, stream)
DEF_FN(cudaError_t, cudaMemcpy2DFromArray, void*, dst, size_t, dpitch, cudaArray_const_t, src, size_t, wOffset, size_t, hOffset, size_t, width, size_t, height, enum cudaMemcpyKind, kind)
DEF_FN(cudaError_t, cudaMemcpy2DFromArrayAsync, void*, dst, size_t, dpitch, cudaArray_const_t, src, size_t, wOffset, size_t, hOffset, size_t, width, size_t, height, enum cudaMemcpyKind, kind, cudaStream_t, stream)
DEF_FN(cudaError_t, cudaMemcpy2DToArray, cudaArray_t, dst, size_t, wOffset, size_t, hOffset, const void*, src, size_t, spitch, size_t, width, size_t, height, enum cudaMemcpyKind, kind)
DEF_FN(cudaError_t, cudaMemcpy2DToArrayAsync, cudaArray_t, dst, size_t, wOffset, size_t, hOffset, const void*, src, size_t, spitch, size_t, width, size_t, height, enum cudaMemcpyKind, kind, cudaStream_t, stream)
DEF_FN(cudaError_t, cudaMemcpy3D, const struct cudaMemcpy3DParms*, p)
DEF_FN(cudaError_t, cudaMemcpy3DAsync, const struct cudaMemcpy3DParms*, p, cudaStream_t, stream)
DEF_FN(cudaError_t, cudaMemcpy3DPeer, const struct cudaMemcpy3DPeerParms*, p)
DEF_FN(cudaError_t, cudaMemcpy3DPeerAsync, const struct cudaMemcpy3DPeerParms*, p, cudaStream_t, stream)

cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    return cudaMemcpy(dst, src, count, kind);
}
DEF_FN(cudaError_t, cudaMemcpyFromSymbol, void*, dst, const void*, symbol, size_t, count, size_t, offset, enum cudaMemcpyKind, kind)
DEF_FN(cudaError_t, cudaMemcpyFromSymbolAsync, void*, dst, const void*, symbol, size_t, count, size_t, offset, enum cudaMemcpyKind, kind, cudaStream_t, stream)
DEF_FN(cudaError_t, cudaMemcpyPeer, void*, dst, int,  dstDevice, const void*, src, int,  srcDevice, size_t, count)
DEF_FN(cudaError_t, cudaMemcpyPeerAsync, void*, dst, int,  dstDevice, const void*, src, int,  srcDevice, size_t, count, cudaStream_t, stream)
cudaError_t cudaMemcpyToSymbol(const void* symbol, const void* src, size_t count, size_t offset, enum cudaMemcpyKind kind)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
    memcpy_cnt += count;
#endif //WITH_API_CNT
    int ret = 1;
    enum clnt_stat retval;
    if (kind == cudaMemcpyHostToDevice) {
        int index = hainfo_getindex((void*)src);
        /* not a cudaHostAlloc'ed memory */
        if (index == -1) {
            mem_data src_mem;
            src_mem.mem_data_len = count;
            src_mem.mem_data_val = (void*)src;
            retval = cuda_memcpy_to_symbol_1((ptr)symbol, src_mem, count, offset, &ret, clnt);
        } else {
            retval = cuda_memcpy_to_symbol_shm_1(index, (ptr)symbol, count, offset, kind, &ret, clnt);
        }
        if (retval != RPC_SUCCESS) {
            clnt_perror (clnt, "call failed");
        }
    } else {
        LOGE(LOG_ERROR, "a kind different from HostToDevice is unsupported for cudaMemcpyToSymbol");
    }
cleanup:
    return ret;
}

DEF_FN(cudaError_t, cudaMemcpyToSymbolAsync, const void*, symbol, const void*, src, size_t, count, size_t, offset, enum cudaMemcpyKind, kind, cudaStream_t, stream)

cudaError_t cudaMemset(void* devPtr, int value, size_t count)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval;
    retval = cuda_memset_1((ptr)devPtr, value, count, &result, clnt);
    if (retval != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    return result;
}

cudaError_t cudaMemset2D(void* devPtr, size_t pitch, int value, size_t width, size_t height)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval;
    retval = cuda_memset_2d_1((ptr)devPtr, pitch, value, width, height, &result, clnt);
    if (retval != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    return result;
}

DEF_FN(cudaError_t, cudaMemset2DAsync, void*, devPtr, size_t, pitch, int,  value, size_t, width, size_t, height, cudaStream_t, stream)

cudaError_t cudaMemset3D(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval;
    retval = cuda_memset_3d_1(pitchedDevPtr.pitch,
                              (ptr)pitchedDevPtr.ptr,
                              pitchedDevPtr.xsize,
                              pitchedDevPtr.ysize,
                              value,
                              extent.depth,
                              extent.height,
                              extent.width, 
                              &result, clnt);
    if (retval != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    return result;
}

DEF_FN(cudaError_t, cudaMemset3DAsync, struct cudaPitchedPtr, pitchedDevPtr, int,  value, struct cudaExtent, extent, cudaStream_t, stream)
DEF_FN(cudaError_t, cudaMemsetAsync, void*, devPtr, int,  value, size_t, count, cudaStream_t, stream)

DEF_FN(struct cudaExtent, make_cudaExtent, size_t, w, size_t, h, size_t, d)
DEF_FN(struct cudaPitchedPtr, make_cudaPitchedPtr, void*, d, size_t, p, size_t, xsz, size_t, ysz)
DEF_FN(struct cudaPos, make_cudaPos, size_t, x, size_t, y, size_t, z)

DEF_FN(cudaError_t, cudaPointerGetAttributes, struct cudaPointerAttributes*, attributes, const void*, ptr)

cudaError_t cudaDeviceCanAccessPeer(int* canAccessPeer, int device, int peerDevice)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int_result result;
    enum clnt_stat retval;
    retval = cuda_device_can_access_peer_1(device,
                                           peerDevice,
                                           &result, clnt);
    if (retval != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    if (result.err == 0) {
        *canAccessPeer = result.int_result_u.data;
    }
    return result.err;
}

cudaError_t cudaDeviceDisablePeerAccess(int peerDevice)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval;
    retval = cuda_device_disable_peer_access_1(
                                           peerDevice,
                                           &result, clnt);
    if (retval != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    return result;
}

cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval;
    retval = cuda_device_disable_peer_access_1(
                                           peerDevice,
                                           &result, clnt);
    if (retval != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    return result;
}

DEF_FN(struct cudaChannelFormatDesc, cudaCreateChannelDesc, int,  x, int,  y, int,  z, int,  w, enum cudaChannelFormatKind, f)
DEF_FN(cudaError_t, cudaCreateTextureObject, cudaTextureObject_t*, pTexObject, const struct cudaResourceDesc*, pResDesc, const struct cudaTextureDesc*, pTexDesc, const struct cudaResourceViewDesc*, pResViewDesc)
DEF_FN(cudaError_t, cudaDestroyTextureObject, cudaTextureObject_t, texObject)
DEF_FN(cudaError_t, cudaGetChannelDesc, struct cudaChannelFormatDesc*, desc, cudaArray_const_t, array)
DEF_FN(cudaError_t, cudaGetTextureObjectResourceDesc, struct cudaResourceDesc*, pResDesc, cudaTextureObject_t, texObject)
DEF_FN(cudaError_t, cudaGetTextureObjectResourceViewDesc, struct cudaResourceViewDesc*, pResViewDesc, cudaTextureObject_t, texObject)
DEF_FN(cudaError_t, cudaGetTextureObjectTextureDesc,
       struct cudaTextureDesc*, pTexDesc, cudaTextureObject_t, texObject)
DEF_FN(cudaError_t, cudaCreateSurfaceObject, cudaSurfaceObject_t*, pSurfObject, const struct cudaResourceDesc*, pResDesc)
DEF_FN(cudaError_t, cudaDestroySurfaceObject, cudaSurfaceObject_t, surfObject)
DEF_FN(cudaError_t, cudaGetSurfaceObjectResourceDesc, struct cudaResourceDesc*, pResDesc, cudaSurfaceObject_t, surfObject)

cudaError_t cudaDriverGetVersion(int* driverVersion)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int_result result;
    enum clnt_stat retval;
    retval = cuda_driver_get_version_1(&result, clnt);
    if (retval != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    if (result.err == 0) {
        *driverVersion = result.int_result_u.data;
    }
    return result.err;
}

cudaError_t cudaRuntimeGetVersion(int* runtimeVersion)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int_result result;
    enum clnt_stat retval;
    retval = cuda_runtime_get_version_1(&result, clnt);
    if (retval != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    if (result.err == 0) {
        *runtimeVersion = result.int_result_u.data;
    }
    return result.err;
}

DEF_FN(cudaError_t, cudaGraphAddChildGraphNode, cudaGraphNode_t*, pGraphNode, cudaGraph_t, graph, const cudaGraphNode_t*, pDependencies, size_t, numDependencies, cudaGraph_t, childGraph)
DEF_FN(cudaError_t, cudaGraphAddDependencies, cudaGraph_t, graph, const cudaGraphNode_t*, from, const cudaGraphNode_t*, to, size_t, numDependencies)
DEF_FN(cudaError_t, cudaGraphAddEmptyNode, cudaGraphNode_t*, pGraphNode, cudaGraph_t, graph, const cudaGraphNode_t*, pDependencies, size_t, numDependencies)
DEF_FN(cudaError_t, cudaGraphAddHostNode, cudaGraphNode_t*, pGraphNode, cudaGraph_t, graph, const cudaGraphNode_t*, pDependencies, size_t, numDependencies, const struct cudaHostNodeParams*, pNodeParams)
DEF_FN(cudaError_t, cudaGraphAddKernelNode, cudaGraphNode_t*, pGraphNode, cudaGraph_t, graph, const cudaGraphNode_t*, pDependencies, size_t, numDependencies, const struct cudaKernelNodeParams*, pNodeParams)
DEF_FN(cudaError_t, cudaGraphAddMemcpyNode, cudaGraphNode_t*, pGraphNode, cudaGraph_t, graph, const cudaGraphNode_t*, pDependencies, size_t, numDependencies, const struct cudaMemcpy3DParms*, pCopyParams)
DEF_FN(cudaError_t, cudaGraphAddMemsetNode, cudaGraphNode_t*, pGraphNode, cudaGraph_t, graph, const cudaGraphNode_t*, pDependencies, size_t, numDependencies, const struct cudaMemsetParams*, pMemsetParams)
DEF_FN(cudaError_t, cudaGraphChildGraphNodeGetGraph, cudaGraphNode_t, node, cudaGraph_t*, pGraph)
DEF_FN(cudaError_t, cudaGraphClone, cudaGraph_t*, pGraphClone, cudaGraph_t, originalGraph)
DEF_FN(cudaError_t, cudaGraphCreate, cudaGraph_t*, pGraph, unsigned int, flags)
DEF_FN(cudaError_t, cudaGraphDestroy, cudaGraph_t, graph)
DEF_FN(cudaError_t, cudaGraphDestroyNode, cudaGraphNode_t, node)
DEF_FN(cudaError_t, cudaGraphExecDestroy, cudaGraphExec_t, graphExec)
DEF_FN(cudaError_t, cudaGraphExecKernelNodeSetParams, cudaGraphExec_t, hGraphExec, cudaGraphNode_t, node, const struct cudaKernelNodeParams*, pNodeParams)
DEF_FN(cudaError_t, cudaGraphGetEdges, cudaGraph_t, graph, cudaGraphNode_t*, from, cudaGraphNode_t*, to, size_t*, numEdges)
DEF_FN(cudaError_t, cudaGraphGetNodes, cudaGraph_t, graph, cudaGraphNode_t*, nodes, size_t*, numNodes)
DEF_FN(cudaError_t, cudaGraphGetRootNodes, cudaGraph_t, graph, cudaGraphNode_t*, pRootNodes, size_t*, pNumRootNodes)
DEF_FN(cudaError_t, cudaGraphHostNodeGetParams, cudaGraphNode_t, node, struct cudaHostNodeParams*, pNodeParams)
DEF_FN(cudaError_t, cudaGraphHostNodeSetParams, cudaGraphNode_t, node, const struct cudaHostNodeParams*, pNodeParams)
DEF_FN(cudaError_t, cudaGraphInstantiate, cudaGraphExec_t*, pGraphExec, cudaGraph_t, graph, cudaGraphNode_t*, pErrorNode, char*, pLogBuffer, size_t, bufferSize)
DEF_FN(cudaError_t, cudaGraphKernelNodeGetParams, cudaGraphNode_t, node, struct cudaKernelNodeParams*, pNodeParams)
DEF_FN(cudaError_t, cudaGraphKernelNodeSetParams, cudaGraphNode_t, node, const struct cudaKernelNodeParams*, pNodeParams)
DEF_FN(cudaError_t, cudaGraphLaunch, cudaGraphExec_t, graphExec, cudaStream_t, stream)
DEF_FN(cudaError_t, cudaGraphMemcpyNodeGetParams, cudaGraphNode_t, node, struct cudaMemcpy3DParms*, pNodeParams)
DEF_FN(cudaError_t, cudaGraphMemcpyNodeSetParams, cudaGraphNode_t, node, const struct cudaMemcpy3DParms*, pNodeParams)
DEF_FN(cudaError_t, cudaGraphMemsetNodeGetParams, cudaGraphNode_t, node, struct cudaMemsetParams*, pNodeParams)
DEF_FN(cudaError_t, cudaGraphMemsetNodeSetParams, cudaGraphNode_t, node, const struct cudaMemsetParams*, pNodeParams)
DEF_FN(cudaError_t, cudaGraphNodeFindInClone, cudaGraphNode_t*, pNode, cudaGraphNode_t, originalNode, cudaGraph_t, clonedGraph)
DEF_FN(cudaError_t, cudaGraphNodeGetDependencies, cudaGraphNode_t, node, cudaGraphNode_t*, pDependencies, size_t*, pNumDependencies)
DEF_FN(cudaError_t, cudaGraphNodeGetDependentNodes, cudaGraphNode_t, node, cudaGraphNode_t*, pDependentNodes, size_t*, pNumDependentNodes)
DEF_FN(cudaError_t, cudaGraphNodeGetType, cudaGraphNode_t, node, enum cudaGraphNodeType*, pType)
DEF_FN(cudaError_t, cudaGraphRemoveDependencies, cudaGraph_t, graph, const cudaGraphNode_t*, from, const cudaGraphNode_t*, to, size_t, numDependencies)
DEF_FN(cudaError_t, cudaProfilerInitialize, const char*, configFile, const char*, outputFile, cudaOutputMode_t, outputMode)
DEF_FN(cudaError_t, cudaProfilerStart, void)
DEF_FN(cudaError_t, cudaProfilerStop, void)

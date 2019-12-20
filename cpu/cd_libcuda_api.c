#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <driver_types.h>
#include <surface_types.h>
#include <texture_types.h>
#include <cuda_runtime_api.h>

#include "cd_libwrap.h"
#include "cd_rpc_prot.h"
#include "cd_common.h"

//static const char* LIBCUDA_PATH = "/lib64/libcuda.so";
static const char* LIBCUDA_PATH = "/usr/local/cuda/lib64/libcudart.so";
static void *so_handle = NULL;

static CLIENT *clnt = NULL;

void __attribute__ ((constructor)) init_rpc(void)
{
    enum clnt_stat retval_1;
    int result_1;
    int_result result_2;
    char *printmessage_1_arg1 = "hello";
    struct sockaddr_un sock = {.sun_family = AF_UNIX,
                               .sun_path = CD_SOCKET_PATH};
    int isock = RPC_ANYSOCK;
    printf("test\n");
    clnt = clntunix_create(&sock, RPC_CD_PROG, RPC_CD_VERS, &isock, 0, 0);
    if (clnt == NULL) {
        printf("error\n");
        exit (1);
    }

    retval_1 = printmessage_1(printmessage_1_arg1, &result_1, clnt);
    printf("return:%d\n", result_1);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
}

static void deinit_rpc(void)
{
    clnt_destroy (clnt);
}

static inline void* libwrap_get_sohandle()
{
    if (!so_handle) {
        if ( !(so_handle = dlopen(LIBCUDA_PATH, RTLD_LAZY)) ) {
            fprintf(stderr, "%s\n", dlerror());
            so_handle = NULL;
            return 0;
        }
    }
    return so_handle;
}

static inline void libwrap_pre_call(char *ret, char *name, char *parameters)
{
    printf("%s\n", name);
}
static inline void libwrap_post_call(char *ret, char *name, char *parameters)
{
    printf("%s\n", name);
}


DEF_FN(cudaError_t, cudaChooseDevice, int*, device, const struct cudaDeviceProp*, prop)
DEF_FN(cudaError_t, cudaDeviceGetAttribute, int*, value, enum cudaDeviceAttr, attr, int, device)
DEF_FN(cudaError_t, cudaDeviceGetByPCIBusId, int*, device, const char*, pciBusId)
DEF_FN(cudaError_t, cudaDeviceGetCacheConfig, enum cudaFuncCache*, pCacheConfig)
DEF_FN(cudaError_t, cudaDeviceGetLimit, size_t*, pValue, enum cudaLimit, limit)
DEF_FN(cudaError_t, cudaDeviceGetP2PAttribute, int*, value, enum cudaDeviceP2PAttr, attr, int,  srcDevice, int,  dstDevice)
DEF_FN(cudaError_t, cudaDeviceGetPCIBusId, char*, pciBusId, int, len, int, device)
DEF_FN(cudaError_t, cudaDeviceGetSharedMemConfig, enum cudaSharedMemConfig*, pConfig)
DEF_FN(cudaError_t, cudaDeviceGetStreamPriorityRange, int*, leastPriority, int*, greatestPriority)
DEF_FN(cudaError_t, cudaDeviceReset, void)
DEF_FN(cudaError_t, cudaDeviceSetCacheConfig, enum cudaFuncCache, cacheConfig)
DEF_FN(cudaError_t, cudaDeviceSetLimit, enum cudaLimit, limit, size_t, value)
DEF_FN(cudaError_t, cudaDeviceSetSharedMemConfig, enum cudaSharedMemConfig, config)
DEF_FN(cudaError_t, cudaDeviceSynchronize, void)
DEF_FN(cudaError_t, cudaGetDevice, int*, device)
DEF_FN(cudaError_t, cudaGetDeviceCount, int*, count)
DEF_FN(cudaError_t, cudaGetDeviceFlags, unsigned int*, flags)
DEF_FN(cudaError_t, cudaGetDeviceProperties, struct cudaDeviceProp*, prop, int,  device)
DEF_FN(cudaError_t, cudaIpcCloseMemHandle, void*, devPtr)
DEF_FN(cudaError_t, cudaIpcGetEventHandle, cudaIpcEventHandle_t*, handle, cudaEvent_t, event)
DEF_FN(cudaError_t, cudaIpcGetMemHandle, cudaIpcMemHandle_t*, handle, void*, devPtr)
DEF_FN(cudaError_t, cudaIpcOpenEventHandle, cudaEvent_t*, event, cudaIpcEventHandle_t, handle)
DEF_FN(cudaError_t, cudaIpcOpenMemHandle, void**, devPtr, cudaIpcMemHandle_t, handle, unsigned int,  flags)
DEF_FN(cudaError_t, cudaSetDevice, int,  device)
DEF_FN(cudaError_t, cudaSetDeviceFlags, unsigned int,  flags)
DEF_FN(cudaError_t, cudaSetValidDevices, int*, device_arr, int,  len)
DEF_FN(const char*, cudaGetErrorName, cudaError_t, error)
DEF_FN(const char*, cudaGetErrorString, cudaError_t, error)
DEF_FN(cudaError_t, cudaGetLastError, void)
DEF_FN(cudaError_t, cudaPeekAtLastError, void)
DEF_FN(cudaError_t, cudaStreamAddCallback, cudaStream_t, stream, cudaStreamCallback_t, callback, void*, userData, unsigned int,  flags)
DEF_FN(cudaError_t, cudaStreamAttachMemAsync, cudaStream_t, stream, void*, devPtr, size_t, length, unsigned int,  flags)
DEF_FN(cudaError_t, cudaStreamBeginCapture, cudaStream_t, stream, enum cudaStreamCaptureMode, mode)
DEF_FN(cudaError_t, cudaStreamCreate, cudaStream_t*, pStream)
DEF_FN(cudaError_t, cudaStreamCreateWithFlags, cudaStream_t*, pStream, unsigned int,  flags)
DEF_FN(cudaError_t, cudaStreamCreateWithPriority, cudaStream_t*, pStream, unsigned int,  flags, int,  priority)
DEF_FN(cudaError_t, cudaStreamDestroy, cudaStream_t, stream)
DEF_FN(cudaError_t, cudaStreamEndCapture, cudaStream_t, stream, cudaGraph_t*, pGraph)
DEF_FN(cudaError_t, cudaStreamGetCaptureInfo, cudaStream_t, stream, cudaStreamCaptureStatus**, pCaptureStatus, unsigned long, long*, pId)
DEF_FN(cudaError_t, cudaStreamGetFlags, cudaStream_t, hStream, unsigned int*, flags)
DEF_FN(cudaError_t, cudaStreamGetPriority, cudaStream_t, hStream, int*, priority)
DEF_FN(cudaError_t, cudaStreamIsCapturing, cudaStream_t, stream, enum cudaStreamCaptureStatus*, pCaptureStatus)
DEF_FN(cudaError_t, cudaStreamQuery, cudaStream_t, stream)
DEF_FN(cudaError_t, cudaStreamSynchronize, cudaStream_t, stream)
DEF_FN(cudaError_t, cudaStreamWaitEvent, cudaStream_t, stream, cudaEvent_t, event, unsigned int,  flags)
DEF_FN(cudaError_t, cudaThreadExchangeStreamCaptureMode, enum cudaStreamCaptureMode*, mode)
DEF_FN(cudaError_t, cudaEventCreate, cudaEvent_t*, event)
DEF_FN(cudaError_t, cudaEventCreateWithFlags, cudaEvent_t*, event, unsigned int,  flags)
DEF_FN(cudaError_t, cudaEventDestroy, cudaEvent_t, event)
DEF_FN(cudaError_t, cudaEventElapsedTime, float*, ms, cudaEvent_t, start, cudaEvent_t, end)
DEF_FN(cudaError_t, cudaEventQuery, cudaEvent_t, event)
DEF_FN(cudaError_t, cudaEventRecord, cudaEvent_t, event, cudaStream_t, stream)
DEF_FN(cudaError_t, cudaEventSynchronize, cudaEvent_t, event)
DEF_FN(cudaError_t, cudaDestroyExternalMemory, cudaExternalMemory_t, extMem)
DEF_FN(cudaError_t, cudaDestroyExternalSemaphore, cudaExternalSemaphore_t, extSem)
DEF_FN(cudaError_t, cudaExternalMemoryGetMappedBuffer, void**, devPtr, cudaExternalMemory_t, extMem, const struct cudaExternalMemoryBufferDesc*, bufferDesc)
DEF_FN(cudaError_t, cudaExternalMemoryGetMappedMipmappedArray, cudaMipmappedArray_t*, mipmap, cudaExternalMemory_t, extMem, const struct cudaExternalMemoryMipmappedArrayDesc*, mipmapDesc)
DEF_FN(cudaError_t, cudaImportExternalMemory, cudaExternalMemory_t*, extMem_out, const struct cudaExternalMemoryHandleDesc*, memHandleDesc)
DEF_FN(cudaError_t, cudaImportExternalSemaphore, cudaExternalSemaphore_t*, extSem_out, const struct cudaExternalSemaphoreHandleDesc*, semHandleDesc)
DEF_FN(cudaError_t, cudaSignalExternalSemaphoresAsync, const cudaExternalSemaphore_t*, extSemArray, const struct cudaExternalSemaphoreSignalParams*, paramsArray, unsigned int,  numExtSems, cudaStream_t, stream)
DEF_FN(cudaError_t, cudaWaitExternalSemaphoresAsync, const cudaExternalSemaphore_t*, extSemArray, const struct cudaExternalSemaphoreWaitParams*, paramsArray, unsigned int,  numExtSems, cudaStream_t, stream)
DEF_FN(cudaError_t, cudaFuncGetAttributes, struct cudaFuncAttributes*, attr, const void*, func)
DEF_FN(cudaError_t, cudaFuncSetAttribute, const void*, func, enum cudaFuncAttribute, attr, int, value)
DEF_FN(cudaError_t, cudaFuncSetCacheConfig, const void*, func, enum cudaFuncCache, cacheConfig)
DEF_FN(cudaError_t, cudaFuncSetSharedMemConfig, const void*, func, enum cudaSharedMemConfig, config)
DEF_FN(void* cudaGetParameterBuffer, size_t, alignment, size_t, size)
DEF_FN(void* cudaGetParameterBufferV2, void*, func, dim3, gridDimension, dim3, blockDimension, unsigned int,  sharedMemSize)
DEF_FN(cudaError_t, cudaLaunchCooperativeKernel, const void*, func, dim3, gridDim, dim3, blockDim, void**, args, size_t, sharedMem, cudaStream_t, stream)
DEF_FN(cudaError_t, cudaLaunchCooperativeKernelMultiDevice, struct cudaLaunchParams*, launchParamsList, unsigned int, numDevices, unsigned int, flags)
DEF_FN(cudaError_t, cudaLaunchHostFunc, cudaStream_t, stream, cudaHostFn_t, fn, void*, userData)
//DEF_FN(cudaError_t, cudaLaunchKernel, const void*, func, dim3, gridDim, dim3, blockDim, void**, args, size_t, sharedMem, cudaStream_t, stream)
cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream)
{
    int result;
    enum clnt_stat retval_1;
    rpc_dim3 rpc_gridDim = {gridDim.x, gridDim.y, gridDim.z};
    rpc_dim3 rpc_blockDim = {blockDim.x, blockDim.y, blockDim.z};
    mem_data rpc_args;
    rpc_args.mem_data_len = 0;
    rpc_args.mem_data_val = NULL;
    retval_1 = cuda_launch_kernel_1((uint64_t)func, rpc_gridDim, rpc_blockDim, rpc_args, sharedMem, (uint64_t)stream, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    return result;
}
DEF_FN(cudaError_t, cudaSetDoubleForDevice, double*, d)
DEF_FN(cudaError_t, cudaSetDoubleForHost, double*, d)
DEF_FN(cudaError_t, cudaOccupancyMaxActiveBlocksPerMultiprocessor, int*, numBlocks, const void*, func, int,  blockSize, size_t, dynamicSMemSize)
DEF_FN(cudaError_t, cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags, int*, numBlocks, const void*, func, int,  blockSize, size_t, dynamicSMemSize, unsigned int,  flags)
DEF_FN(cudaError_t, cudaArrayGetInfo, struct cudaChannelFormatDesc*, desc, struct cudaExtent*, extent, unsigned int*, flags, cudaArray_t, array)
DEF_FN(cudaError_t, cudaFree, void*, devPtr)
DEF_FN(cudaError_t, cudaFreeArray, cudaArray_t, array)
DEF_FN(cudaError_t, cudaFreeHost, void*, ptr)
DEF_FN(cudaError_t, cudaFreeMipmappedArray, cudaMipmappedArray_t, mipmappedArray)
DEF_FN(cudaError_t, cudaGetMipmappedArrayLevel, cudaArray_t*, levelArray, cudaMipmappedArray_const_t, mipmappedArray, unsigned int,  level)
DEF_FN(cudaError_t, cudaGetSymbolAddress, void**, devPtr, const void*, symbol)
DEF_FN(cudaError_t, cudaGetSymbolSize, size_t*, size, const void*, symbol)
DEF_FN(cudaError_t, cudaHostAlloc, void**, pHost, size_t, size, unsigned int,  flags)
DEF_FN(cudaError_t, cudaHostGetDevicePointer, void**, pDevice, void*, pHost, unsigned int,  flags)
DEF_FN(cudaError_t, cudaHostGetFlags, unsigned int*, pFlags, void*, pHost)
DEF_FN(cudaError_t, cudaHostRegister, void*, ptr, size_t, size, unsigned int,  flags)
DEF_FN(cudaError_t, cudaHostUnregister, void*, ptr)
//DEF_FN(cudaError_t, cudaMalloc, void**, devPtr, size_t, size)
cudaError_t cudaMalloc(void** devPtr, size_t size)
{
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
DEF_FN(cudaError_t, cudaMalloc3D, struct cudaPitchedPtr*, pitchedDevPtr, struct cudaExtent, extent)
DEF_FN(cudaError_t, cudaMalloc3DArray, cudaArray_t*, array, const struct cudaChannelFormatDesc*, desc, struct cudaExtent, extent, unsigned int,  flags)
DEF_FN(cudaError_t, cudaMallocArray, cudaArray_t*, array, const struct cudaChannelFormatDesc*, desc, size_t, width, size_t, height, unsigned int,  flags)
DEF_FN(cudaError_t, cudaMallocHost, void**, ptr, size_t, size)
DEF_FN(cudaError_t, cudaMallocManaged, void**, devPtr, size_t, size, unsigned int,  flags)
DEF_FN(cudaError_t, cudaMallocMipmappedArray, cudaMipmappedArray_t*, mipmappedArray, const struct cudaChannelFormatDesc*, desc, struct cudaExtent, extent, unsigned int,  numLevels, unsigned int,  flags)
DEF_FN(cudaError_t, cudaMallocPitch, void**, devPtr, size_t*, pitch, size_t, width, size_t, height)
DEF_FN(cudaError_t, cudaMemAdvise, const void*, devPtr, size_t, count, enum cudaMemoryAdvise, advice, int,  device)
DEF_FN(cudaError_t, cudaMemGetInfo, size_t*, free, size_t*, total)
DEF_FN(cudaError_t, cudaMemPrefetchAsync, const void*, devPtr, size_t, count, int,  dstDevice, cudaStream_t, stream)
DEF_FN(cudaError_t, cudaMemRangeGetAttribute, void*, data, size_t, dataSize, enum cudaMemRangeAttribute, attribute, const void*, devPtr, size_t, count)
DEF_FN(cudaError_t, cudaMemRangeGetAttributes, void**, data, size_t*, dataSizes, enum cudaMemRangeAttribute*, attributes, size_t, numAttributes, const void*, devPtr, size_t, count)
//DEF_FN(cudaError_t, cudaMemcpy, void*, dst, const void*, src, size_t, count, enum cudaMemcpyKind, kind)
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind)
{
    if (kind == cudaMemcpyHostToDevice) {
        int result;
        enum clnt_stat retval;
        mem_data src_mem;
        src_mem.mem_data_len = count;
        src_mem.mem_data_val = (void*)src;
        retval = cuda_memcpy_htod_1((uint64_t)dst, src_mem, count, &result, clnt);
        if (retval != RPC_SUCCESS) {
            clnt_perror (clnt, "call failed");
        }
        return result;
    } else if (kind == cudaMemcpyDeviceToHost) {
        mem_result result;
        enum clnt_stat retval;
        retval = cuda_memcpy_dtoh_1((uint64_t)src, count, &result, clnt);
        if (retval != RPC_SUCCESS) {
            clnt_perror (clnt, "call failed");
        }
        if (result.err != 0) {
            return result.err;
        }
        if (result.mem_result_u.data.mem_data_len != count) {
            fprintf(stderr, "error\n");
            return result.err;
        }
        memcpy(dst, (void*)result.mem_result_u.data.mem_data_val, count);
        return result.err;
    } else {
        fprintf(stderr, "error\n");
    }
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
DEF_FN(cudaError_t, cudaMemcpyAsync, void*, dst, const void*, src, size_t, count, enum cudaMemcpyKind, kind, cudaStream_t, stream)
DEF_FN(cudaError_t, cudaMemcpyFromSymbol, void*, dst, const void*, symbol, size_t, count, size_t, offset, enum cudaMemcpyKind, kind)
DEF_FN(cudaError_t, cudaMemcpyFromSymbolAsync, void*, dst, const void*, symbol, size_t, count, size_t, offset, enum cudaMemcpyKind, kind, cudaStream_t, stream)
DEF_FN(cudaError_t, cudaMemcpyPeer, void*, dst, int,  dstDevice, const void*, src, int,  srcDevice, size_t, count)
DEF_FN(cudaError_t, cudaMemcpyPeerAsync, void*, dst, int,  dstDevice, const void*, src, int,  srcDevice, size_t, count, cudaStream_t, stream)
DEF_FN(cudaError_t, cudaMemcpyToSymbol, const void*, symbol, const void*, src, size_t, count, size_t, offset, enum cudaMemcpyKind, kind)
DEF_FN(cudaError_t, cudaMemcpyToSymbolAsync, const void*, symbol, const void*, src, size_t, count, size_t, offset, enum cudaMemcpyKind, kind, cudaStream_t, stream)
DEF_FN(cudaError_t, cudaMemset, void*, devPtr, int,  value, size_t, count)
DEF_FN(cudaError_t, cudaMemset2D, void*, devPtr, size_t, pitch, int,  value, size_t, width, size_t, height)
DEF_FN(cudaError_t, cudaMemset2DAsync, void*, devPtr, size_t, pitch, int,  value, size_t, width, size_t, height, cudaStream_t, stream)
DEF_FN(cudaError_t, cudaMemset3D, struct cudaPitchedPtr, pitchedDevPtr, int,  value, struct cudaExtent, extent)
DEF_FN(cudaError_t, cudaMemset3DAsync, struct cudaPitchedPtr, pitchedDevPtr, int,  value, struct cudaExtent, extent, cudaStream_t, stream)
DEF_FN(cudaError_t, cudaMemsetAsync, void*, devPtr, int,  value, size_t, count, cudaStream_t, stream)
DEF_FN(struct cudaExtent, make_cudaExtent, size_t, w, size_t, h, size_t, d)
DEF_FN(struct cudaPitchedPtr, make_cudaPitchedPtr, void*, d, size_t, p, size_t, xsz, size_t, ysz)
DEF_FN(struct cudaPos, make_cudaPos, size_t, x, size_t, y, size_t, z)
DEF_FN(cudaError_t, cudaPointerGetAttributes, struct cudaPointerAttributes*, attributes, const void*, ptr)
DEF_FN(cudaError_t, cudaDeviceCanAccessPeer, int*, canAccessPeer, int,  device, int,  peerDevice)
DEF_FN(cudaError_t, cudaDeviceDisablePeerAccess, int,  peerDevice)
DEF_FN(cudaError_t, cudaDeviceEnablePeerAccess, int,  peerDevice, unsigned int,  flags)
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
DEF_FN(cudaError_t, cudaDriverGetVersion, int*, driverVersion)
DEF_FN(cudaError_t, cudaRuntimeGetVersion, int*, runtimeVersion)
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

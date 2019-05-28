#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cudaGL.h>
#include <cudaEGL.h>
#include <vdpau/vdpau.h>
#include <cudaVDPAU.h>
#include <cudaProfiler.h>

#include <driver_types.h>
#include <string.h>

#include "cd_libwrap.h"
#include "cd_client_hidden.h"
#include "cd_rpc_prot.h"
#include "cd_common.h"

static const char* LIBCUDA_PATH = "/lib64/libcuda.so";
static void *so_handle = NULL;

static CLIENT *clnt = NULL;
static void init_rpc(void)
{
	enum clnt_stat retval_1;
	int result_1;
    int_result result_2;
	char *printmessage_1_arg1 = "hello";
    struct sockaddr_un sock = {.sun_family = AF_UNIX,
                               .sun_path = CD_SOCKET_PATH};
    int isock = RPC_ANYSOCK;
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

static void *(*dlopen_orig)(const char *, int) = NULL;
static int   (*dlclose_orig)(void *) = NULL;
static void *dl_handle = NULL;

void *dlopen(const char *filename, int flag)
{
    if (dlopen_orig == NULL) {
        if ( (dlopen_orig = dlsym(RTLD_NEXT, "dlopen")) == NULL) {
            printf("[dlopen] dlsym failed\n");
        }
    }


    if (filename && strcmp(filename, "libcuda.so.1") == 0) {
        dl_handle = dlopen_orig("/home/eiling/projects/cricket/cricketd/libcudawrap.so", flag);
        if (clnt == NULL) {
            init_rpc();
        }
        return dl_handle;
    } else {
        return dlopen_orig(filename, flag);
    }
}

int dlclose(void *handle)
{
    if (!handle) {
        printf("[dlclose] handle NULL\n");
        return -1;
    } else if (dlclose_orig == NULL) {
        if ( (dlclose_orig = dlsym(RTLD_NEXT, "dlclose")) == NULL) {
            printf("[dlclose] dlsym failed\n");
        }
    }


    // Ignore dlclose call that would close this library
    if (dl_handle == handle) {
        printf("[dlclose] ignore close\n");
        return 0;
    } else {
        return dlclose_orig(handle);
    }

}

DEF_FN(CUresult, cuProfilerInitialize, const char*, configFile, const char*, outputFile, CUoutput_mode, outputMode)
DEF_FN(CUresult, cuProfilerStart)
DEF_FN(CUresult, cuProfilerStop)
DEF_FN(CUresult, cuVDPAUGetDevice, CUdevice*, pDevice, VdpDevice, vdpDevice, VdpGetProcAddress*, vdpGetProcAddress)
DEF_FN(CUresult, cuVDPAUCtxCreate, CUcontext*, pCtx, unsigned int, flags, CUdevice, device, VdpDevice, vdpDevice, VdpGetProcAddress*, vdpGetProcAddress)
DEF_FN(CUresult, cuGraphicsVDPAURegisterVideoSurface, CUgraphicsResource*, pCudaResource, VdpVideoSurface, vdpSurface, unsigned int, flags)
DEF_FN(CUresult, cuGraphicsVDPAURegisterOutputSurface, CUgraphicsResource*, pCudaResource, VdpOutputSurface, vdpSurface, unsigned int, flags)

DEF_FN(CUresult, cuDeviceTotalMem, size_t*, bytes, CUdevice, dev)
//DEF_FN(CUresult, cuCtxCreate, CUcontext*, pctx, unsigned int, flags, CUdevice, dev)

CUresult cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev)
{
    DEF_FN_PTR(CUresult, CUcontext*, unsigned int, CUdevice);
    DEF_DLSYM(CUresult, cuCtxCreate)
    CAL_FN_PTR(pctx, flags, dev);
    printf("%s(%p, %u, %d) = %d\n", __FUNCTION__, pctx, flags, dev, ret);
    return ret;
}
DEF_FN(CUresult, cuCtxSynchronize)
DEF_FN(CUresult, cuModuleGetGlobal, CUdeviceptr*, dptr, size_t*, bytes, CUmodule, hmod, const char*, name)
DEF_FN(CUresult, cuMemGetInfo, size_t*, free, size_t*, total)
//DEF_FN(CUresult, cuMemAlloc, CUdeviceptr*, dptr, size_t, bytesize)
CUresult cuMemAlloc(CUdeviceptr* dptr, size_t bytesize)
{
    DEF_FN_PTR(CUresult, CUdeviceptr*, size_t);
    DEF_DLSYM(CUresult, cuMemAlloc_v2)
    printf("pre %s(%p->%p, %lu) = %d\n", __FUNCTION__, dptr, *dptr, bytesize, ret);
    CAL_FN_PTR(dptr, bytesize);
    printf("post %s(%p->%p, %lu) = %d\n", __FUNCTION__, dptr, *dptr, bytesize, ret);
    return ret;
}
DEF_FN(CUresult, cuMemAllocPitch, CUdeviceptr*, dptr, size_t*, pPitch, size_t, WidthInBytes, size_t, Height, unsigned int, ElementSizeBytes)
DEF_FN(CUresult, cuMemFree, CUdeviceptr, dptr)
DEF_FN(CUresult, cuMemGetAddressRange, CUdeviceptr*, pbase, size_t*, psize, CUdeviceptr, dptr)
DEF_FN(CUresult, cuMemHostGetDevicePointer, CUdeviceptr*, pdptr, void*, p, unsigned int, Flags)
DEF_FN(CUresult, cuMemHostRegister, void*, p, size_t, bytesize, unsigned int, Flags)
DEF_FN(CUresult, cuMemsetD8, CUdeviceptr, dstDevice, unsigned char, uc, size_t, N);
DEF_FN(CUresult, cuMemsetD8_v2_ptds, CUdeviceptr, dstDevice, unsigned char, uc, size_t, N);
DEF_FN(CUresult, cuMemsetD2D8, CUdeviceptr, dstDevice, size_t, dstPitch, unsigned char, uc, size_t, Width, size_t, Height)
DEF_FN(CUresult, cuMemsetD2D8_v2_ptds, CUdeviceptr, dstDevice, size_t, dstPitch, unsigned char, uc, size_t, Width, size_t, Height)
DEF_FN(CUresult, cuEventDestroy, CUevent, hEvent)
DEF_FN(CUresult, cuStreamDestroy, CUstream, hStream)
DEF_FN(CUresult, cuGLCtxCreate, CUcontext*, pCtx, unsigned int, Flags, CUdevice, device)
DEF_FN(CUresult, cuArrayCreate, CUarray*, pHandle, const CUDA_ARRAY_DESCRIPTOR*, pAllocateArray)
DEF_FN(CUresult, cuArrayGetDescriptor, CUDA_ARRAY_DESCRIPTOR*, pArrayDescriptor, CUarray, hArray)
DEF_FN(CUresult, cuArray3DCreate, CUarray*, pHandle, const CUDA_ARRAY3D_DESCRIPTOR*, pAllocateArray)
DEF_FN(CUresult, cuArray3DGetDescriptor, CUDA_ARRAY3D_DESCRIPTOR*, pArrayDescriptor, CUarray, hArray)
DEF_FN(CUresult, cuTexRefSetAddress2D, CUtexref, hTexRef, const CUDA_ARRAY_DESCRIPTOR*, desc, CUdeviceptr, dptr, size_t, Pitch)
DEF_FN(CUresult, cuTexRefSetAddress, size_t*, ByteOffset, CUtexref, hTexRef, CUdeviceptr, dptr, size_t, bytes)






DEF_FN(CUresult, cuGLInit)
#undef cuGLGetDevices
#undef cuGLMapBufferObject_v2
#undef cuGLMapBufferObjectAsync_v2
DEF_FN(CUresult, cuGLGetDevices, unsigned int*, pCudaDeviceCount, CUdevice*, pCudaDevices, unsigned int, cudaDeviceCount, CUGLDeviceList, deviceList)
DEF_FN(CUresult, cuGLRegisterBufferObject, GLuint, buffer)
DEF_FN(CUresult, cuGLMapBufferObject_v2, CUdeviceptr*, dptr, size_t*, size, GLuint, buffer)
DEF_FN(CUresult, cuGLMapBufferObject_v2_ptds, CUdeviceptr*, dptr, size_t*, size, GLuint, buffer)
DEF_FN(CUresult, cuGLMapBufferObjectAsync_v2, CUdeviceptr*, dptr, size_t*, size, GLuint, buffer, CUstream, hStream)
DEF_FN(CUresult, cuGLMapBufferObjectAsync_v2_ptsz, CUdeviceptr*, dptr, size_t*, size, GLuint, buffer, CUstream, hStream)
DEF_FN(CUresult, cuGLUnmapBufferObject, GLuint, buffer)
DEF_FN(CUresult, cuGLUnmapBufferObjectAsync, GLuint, buffer, CUstream, hStream)
DEF_FN(CUresult, cuGLUnregisterBufferObject, GLuint, buffer)
DEF_FN(CUresult, cuGLSetBufferObjectMapFlags, GLuint, buffer, unsigned int, Flags)
DEF_FN(CUresult, cuGraphicsGLRegisterImage, CUgraphicsResource*, pCudaResource, GLuint, image, GLenum, target, unsigned int, Flags)
DEF_FN(CUresult, cuGraphicsGLRegisterBuffer, CUgraphicsResource*, pCudaResource, GLuint, buffer, unsigned int, Flags)
DEF_FN(CUresult, cuGraphicsEGLRegisterImage, CUgraphicsResource*, pCudaResource, EGLImageKHR, image, unsigned int, flags)
DEF_FN(CUresult, cuEGLStreamConsumerConnect, CUeglStreamConnection*, conn, EGLStreamKHR, stream)
DEF_FN(CUresult, cuEGLStreamConsumerDisconnect, CUeglStreamConnection*, conn)
DEF_FN(CUresult, cuEGLStreamConsumerAcquireFrame, CUeglStreamConnection*, conn, CUgraphicsResource*, pCudaResource, CUstream*, pStream, unsigned int, timeout)
DEF_FN(CUresult, cuEGLStreamConsumerReleaseFrame, CUeglStreamConnection*, conn, CUgraphicsResource, pCudaResource, CUstream*, pStream)
DEF_FN(CUresult, cuEGLStreamProducerConnect, CUeglStreamConnection*, conn, EGLStreamKHR, stream, EGLint, width, EGLint, height)
DEF_FN(CUresult, cuEGLStreamProducerDisconnect, CUeglStreamConnection*, conn)
DEF_FN(CUresult, cuEGLStreamProducerPresentFrame, CUeglStreamConnection*, conn, CUeglFrame, eglframe, CUstream*, pStream)
DEF_FN(CUresult, cuEGLStreamProducerReturnFrame, CUeglStreamConnection*, conn, CUeglFrame*, eglframe, CUstream*, pStream)
DEF_FN(CUresult, cuGraphicsResourceGetMappedEglFrame, CUeglFrame*, eglFrame, CUgraphicsResource, resource, unsigned int, index, unsigned int, mipLevel)
DEF_FN(CUresult, cuEGLStreamConsumerConnectWithFlags, CUeglStreamConnection*, conn, EGLStreamKHR, unsigned int, flags)

//DEF_FN(CUresult, cuInit, unsigned int, Flags)
CUresult cuInit(unsigned int Flags)
{
	enum clnt_stat retval;
    int result;
    retval = rpc_cuinit_1(Flags, &result, clnt);
    printf("[rpc] %s = %d\n", __FUNCTION__, result);
	if (retval != RPC_SUCCESS) {
		fprintf(stderr, "[rpc] %s failed.", __FUNCTION__);
        return CUDA_ERROR_UNKNOWN;
	}
    return result;
}

//DEF_FN(CUresult, cuDeviceGet, CUdevice*, device, int, ordinal)
CUresult cuDeviceGet(CUdevice* device, int ordinal)
{
	enum clnt_stat retval;
    int_result result;
    retval = rpc_cudeviceget_1(ordinal, &result, clnt);
    printf("[rpc] %s = %d, result %d\n", __FUNCTION__, result.err,
                                        result.int_result_u.data);
	if (retval != RPC_SUCCESS) {
		fprintf(stderr, "[rpc] %s failed.", __FUNCTION__);
        return CUDA_ERROR_UNKNOWN;
	}
    *device = result.int_result_u.data;
    return result.err;
}

//DEF_FN(CUresult, cuDeviceGetCount, int*, count)
CUresult cuDeviceGetCount(int* count)
{
	enum clnt_stat retval;
    int_result result;
    retval = rpc_cudevicegetcount_1(&result, clnt);
    printf("[rpc] %s = %d, result %d\n", __FUNCTION__, result.err,
                                        result.int_result_u.data);
	if (retval != RPC_SUCCESS) {
		fprintf(stderr, "[rpc] %s failed.", __FUNCTION__);
        return CUDA_ERROR_UNKNOWN;
	}
    *count = result.int_result_u.data;
    return result.err;
}

DEF_FN(CUresult, cuDeviceGetName, char*, name, int, len, CUdevice, dev)
//DEF_FN(CUresult, cuDeviceGetUuid, CUuuid*, uuid, CUdevice, dev)

/* CUuuid = struct { char bytes[16] };
 * CUdevice = int
 */
CUresult cuDeviceGetUuid(CUuuid* uuid, CUdevice dev)
{
	enum clnt_stat retval;
    uuid_result result;
    retval = rpc_cudevicegetuuid_1(dev, &result, clnt);
    printf("[rpc] %s = %d, result (uuid)\n", __FUNCTION__, result.err);

	if (retval != RPC_SUCCESS) {
		fprintf(stderr, "[rpc] %s failed.", __FUNCTION__);
        return CUDA_ERROR_UNKNOWN;
	}
    memcpy(uuid->bytes, result.uuid_result_u.bytes, 16);
    return result.err;
}

DEF_FN(CUresult, cuDeviceGetLuid, char*, luid, unsigned int*, deviceNodeMask, CUdevice, dev)
DEF_FN(CUresult, cuDeviceGetAttribute, int*, pi, CUdevice_attribute, attrib, CUdevice, dev)
DEF_FN(CUresult, cuDeviceGetProperties, CUdevprop*, prop, CUdevice, dev)
DEF_FN(CUresult, cuDeviceGetByPCIBusId, CUdevice*, dev, const char*, pciBusId)
DEF_FN(CUresult, cuDeviceGetP2PAttribute, int*, value, CUdevice_P2PAttribute, attrib, CUdevice, srcDevice, CUdevice, dstDevice)
//DEF_FN(CUresult, cuDriverGetVersion, int*, driverVersion)
CUresult cuDriverGetVersion(int* driverVersion)
{
	enum clnt_stat retval;
    int_result result;
    retval = rpc_cudrivergetversion_1(&result, clnt);
    printf("[rpc] %s = %d, result %d\n", __FUNCTION__, result.err, result.int_result_u.data);
	if (retval != RPC_SUCCESS) {
		fprintf(stderr, "[rpc] %s failed.", __FUNCTION__);
        return CUDA_ERROR_UNKNOWN;
	}
    *driverVersion = result.int_result_u.data;
    return result.err;
}

DEF_FN(CUresult, cuDeviceGetPCIBusId, char*, pciBusId, int, len, CUdevice, dev)
//DEF_FN(CUresult, cuDevicePrimaryCtxRetain, CUcontext*, pctx, CUdevice, dev)
CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev)
{
    void* real_ctx = *pctx;
    DEF_FN_PTR(CUresult, CUcontext*, CUdevice);
    DEF_DLSYM(CUresult, cuDevicePrimaryCtxRetain)
    CAL_FN_PTR((struct CUctx_st**)&real_ctx, dev);
    if (!(*pctx = cd_client_get_fake_ctx(real_ctx))) {
        fprintf(stderr, "%s: error while translateing ctx\n",
                        __FUNCTION__);
        return CUDA_ERROR_UNKNOWN;
    }
    printf("%s(%p->%p, %d) = %d\n", __FUNCTION__, pctx, *pctx, dev, ret);
    return ret;
}
DEF_FN(CUresult, cuDevicePrimaryCtxRelease, CUdevice, dev)
DEF_FN(CUresult, cuDevicePrimaryCtxSetFlags, CUdevice, dev, unsigned int, flags)
DEF_FN(CUresult, cuDevicePrimaryCtxGetState, CUdevice, dev, unsigned int*, flags, int*, active)
DEF_FN(CUresult, cuDevicePrimaryCtxReset, CUdevice, dev)
DEF_FN(CUresult, cuCtxGetFlags, unsigned int*, flags)
//DEF_FN(CUresult, cuCtxSetCurrent, CUcontext, ctx)
CUresult cuCtxSetCurrent(CUcontext ctx)
{
    void *real_ctx = NULL;
    DEF_FN_PTR(CUresult, CUcontext);
    DEF_DLSYM(CUresult, cuCtxSetCurrent)
    if (!(real_ctx = cd_client_get_real_ctx(ctx))) {
        fprintf(stderr, "%s: error while translating ctx\n",
                        __FUNCTION__);
        return CUDA_ERROR_UNKNOWN;
    }
    CAL_FN_PTR(real_ctx);
    printf("%s(%p->(CUctx_st)) = %d\n", __FUNCTION__, ctx, ret);
    printf("\treal_ctx: %p\n", real_ctx);
    return ret;
}
//DEF_FN(CUresult, cuCtxGetCurrent, CUcontext*, pctx)
CUresult cuCtxGetCurrent(CUcontext *pctx)
{
    void *real_ctx = NULL;
    DEF_FN_PTR(CUresult, CUcontext*);
    DEF_DLSYM(CUresult, cuCtxGetCurrent)
    CAL_FN_PTR((struct CUctx_st**)&real_ctx);
    if (real_ctx != NULL &&
        !(*pctx = cd_client_get_fake_ctx(real_ctx))) {
        fprintf(stderr, "%s: error while translating ctx\n",
                        __FUNCTION__);
        return CUDA_ERROR_UNKNOWN;
    }
    printf("%s(%p->%p) = %d\n", __FUNCTION__, pctx, *pctx, ret);
    return ret;
}
DEF_FN(CUresult, cuCtxDetach, CUcontext, ctx)
DEF_FN(CUresult, cuCtxGetApi2Version, CUcontext, ctx, unsigned int*, version)
//DEF_FN(CUresult, cuCtxGetDevice, CUdevice*, device)
CUresult cuCtxGetDevice(CUdevice *device)
{
    DEF_FN_PTR(CUresult, CUdevice*);
    DEF_DLSYM(CUresult, cuCtxGetDevice) \
    CAL_FN_PTR(device); \
    printf("%s(%p->%d) = %d\n", __FUNCTION__, device, *device, ret);
    return ret;
}
DEF_FN(CUresult, cuCtxGetLimit, size_t*, pvalue, CUlimit, limit)
DEF_FN(CUresult, cuCtxSetLimit, CUlimit, limit, size_t, value)
DEF_FN(CUresult, cuCtxGetCacheConfig, CUfunc_cache*, pconfig)
DEF_FN(CUresult, cuCtxSetCacheConfig, CUfunc_cache, config)
DEF_FN(CUresult, cuCtxGetSharedMemConfig, CUsharedconfig*, pConfig)
DEF_FN(CUresult, cuCtxGetStreamPriorityRange, int*, leastPriority, int*, greatestPriority)
DEF_FN(CUresult, cuCtxSetSharedMemConfig, CUsharedconfig, config)
DEF_FN(CUresult, cuCtxSynchronize, void)
DEF_FN(CUresult, cuModuleLoad, CUmodule*, module, const char*, fname)
DEF_FN(CUresult, cuModuleLoadData, CUmodule*, module, const void*, image)
DEF_FN(CUresult, cuModuleLoadDataEx, CUmodule*, module, const void*, image, unsigned int, numOptions, CUjit_option*, options, void**, optionValues)
DEF_FN(CUresult, cuModuleLoadFatBinary, CUmodule*, module, const void*, fatCubin)
DEF_FN(CUresult, cuModuleUnload, CUmodule, hmod)
//DEF_FN(CUresult, cuModuleGetFunction, CUfunction*, hfunc, CUmodule, hmod, const char*, name)
CUresult cuModuleGetFunction(CUfunction* hfun, CUmodule hmod, const char* name)
{
    void *real_module = NULL;
    DEF_FN_PTR(CUresult, CUfunction*, CUmodule, const char*);
    DEF_DLSYM(CUresult, cuModuleGetFunction)
    printf("pre %s(%p->%p, %p, %s) = %d\n", __FUNCTION__, hfun, *hfun, hmod, name, ret);

    if (!(real_module = cd_client_get_real_module(hmod))) {
        fprintf(stderr, "%s: error while translating module\n",
                        __FUNCTION__);
        return CUDA_ERROR_UNKNOWN;
    }
    CAL_FN_PTR(hfun, real_module, name);
    printf("post %s(%p->%p, %p, %s) = %d\n", __FUNCTION__, hfun, *hfun, hmod, name, ret);
    printf("\treal_module: %p\n", real_module);
    return ret;
}
DEF_FN(CUresult, cuModuleGetTexRef, CUtexref*, pTexRef, CUmodule, hmod, const char*, name)
DEF_FN(CUresult, cuModuleGetSurfRef, CUsurfref*, pSurfRef, CUmodule, hmod, const char*, name)
#undef cuLinkCreate
DEF_FN(CUresult, cuLinkCreate, unsigned int, numOptions, CUjit_option*, options, void**, optionValues, CUlinkState*, stateOut)
#undef cuLinkAddData
DEF_FN(CUresult, cuLinkAddData, CUlinkState, state, CUjitInputType, type, void*, data, size_t, size, const char*, name, unsigned int, numOptions, CUjit_option*, options, void**, optionValues)
#undef cuLinkAddFile
DEF_FN(CUresult, cuLinkAddFile, CUlinkState, state, CUjitInputType, type, const char*, path, unsigned int, numOptions, CUjit_option*, options, void**, optionValues)
DEF_FN(CUresult, cuLinkComplete, CUlinkState, state, void**, cubinOut, size_t*, sizeOut)
DEF_FN(CUresult, cuLinkDestroy, CUlinkState, state)
DEF_FN(CUresult, cuMemAllocManaged, CUdeviceptr*, dptr, size_t, bytesize, unsigned int, flags)
DEF_FN(CUresult, cuMemFreeHost, void*, p)
DEF_FN(CUresult, cuMemHostAlloc, void**, pp, size_t, bytesize, unsigned int, Flags)
DEF_FN(CUresult, cuMemHostGetFlags, unsigned int*, pFlags, void*, p)
DEF_FN(CUresult, cuMemHostUnregister, void*, p)
DEF_FN(CUresult, cuPointerGetAttribute, void*, data, CUpointer_attribute, attribute, CUdeviceptr, ptr)
DEF_FN(CUresult, cuPointerGetAttributes, unsigned int, numAttributes, CUpointer_attribute*, attributes, void**, data, CUdeviceptr, ptr)
DEF_FN(CUresult, cuMemcpy, CUdeviceptr, dst, CUdeviceptr, src, size_t, ByteCount)
DEF_FN(CUresult, cuMemcpy_ptds, CUdeviceptr, dst, CUdeviceptr, src, size_t, ByteCount)
DEF_FN(CUresult, cuMemcpyHtoD, CUdeviceptr, dstDevice, const void*, srcHost, size_t, ByteCount)
DEF_FN(CUresult, cuMemcpyHtoD_v2_ptds, CUdeviceptr, dstDevice, const void*, srcHost, size_t, ByteCount)
DEF_FN(CUresult, cuMemcpyDtoH, void*, dstHost, CUdeviceptr, srcDevice, size_t, ByteCount)
DEF_FN(CUresult, cuMemcpyDtoH_v2_ptds, void*, dstHost, CUdeviceptr, srcDevice, size_t, ByteCount)
DEF_FN(CUresult, cuMemcpyDtoD, CUdeviceptr, dstDevice, CUdeviceptr, srcDevice, size_t, ByteCount)
DEF_FN(CUresult, cuMemcpyDtoD_v2_ptds, CUdeviceptr, dstDevice, CUdeviceptr, srcDevice, size_t, ByteCount)
DEF_FN(CUresult, cuMemcpyDtoA, CUarray, dstArray, size_t, dstOffset, CUdeviceptr, srcDevice, size_t, ByteCount)
DEF_FN(CUresult, cuMemcpyAtoD, CUdeviceptr, dstDevice, CUarray, srcArray, size_t, srcOffset, size_t, ByteCount)
DEF_FN(CUresult, cuMemcpyHtoA, CUarray, dstArray, size_t, dstOffset, const void*, srcHost, size_t, ByteCount)
DEF_FN(CUresult, cuMemcpyAtoH, void*, dstHost, CUarray, srcArray, size_t, srcOffset, size_t, ByteCount)
DEF_FN(CUresult, cuMemcpyAtoA, CUarray, dstArray, size_t, dstOffset, CUarray, srcArray, size_t, srcOffset, size_t, ByteCount)
DEF_FN(CUresult, cuMemcpy2D, const CUDA_MEMCPY2D*, pCopy)
DEF_FN(CUresult, cuMemcpy2DUnaligned, const CUDA_MEMCPY2D*, pCopy)
DEF_FN(CUresult, cuMemcpy2DUnaligned_v2_ptds, const CUDA_MEMCPY2D*, pCopy)
DEF_FN(CUresult, cuMemcpy3D, const CUDA_MEMCPY3D*, pCopy)
DEF_FN(CUresult, cuMemcpy3D_v2_ptds, const CUDA_MEMCPY3D*, pCopy)
DEF_FN(CUresult, cuMemcpyPeerAsync, CUdeviceptr, dstDevice, CUcontext, dstContext, CUdeviceptr, srcDevice, CUcontext, srcContext, size_t, ByteCount, CUstream, hStream)
DEF_FN(CUresult, cuMemcpyPeerAsync_ptsz, CUdeviceptr, dstDevice, CUcontext, dstContext, CUdeviceptr, srcDevice, CUcontext, srcContext, size_t, ByteCount, CUstream, hStream)
DEF_FN(CUresult, cuMemcpyHtoAAsync, CUarray, dstArray, size_t, dstOffset, const void*, srcHost, size_t, ByteCount, CUstream, hStream)
DEF_FN(CUresult, cuMemcpyAtoHAsync, void*, dstHost, CUarray, srcArray, size_t, srcOffset, size_t, ByteCount, CUstream, hStream)
DEF_FN(CUresult, cuMemcpy3DPeerAsync, const CUDA_MEMCPY3D_PEER*, pCopy, CUstream, hStream)
DEF_FN(CUresult, cuMemcpy3DPeerAsync_ptsz, const CUDA_MEMCPY3D_PEER*, pCopy, CUstream, hStream)
DEF_FN(CUresult, cuMemcpyHtoDAsync, CUdeviceptr, dstDevice, const void*, srcHost, size_t, ByteCount, CUstream, hStream)
DEF_FN(CUresult, cuMemcpyHtoDAsync_v2_ptsz, CUdeviceptr, dstDevice, const void*, srcHost, size_t, ByteCount, CUstream, hStream)
DEF_FN(CUresult, cuMemcpyDtoHAsync, void*, dstHost, CUdeviceptr, srcDevice, size_t, ByteCount, CUstream, hStream)
DEF_FN(CUresult, cuMemcpyDtoHAsync_v2_ptsz, void*, dstHost, CUdeviceptr, srcDevice, size_t, ByteCount, CUstream, hStream)
DEF_FN(CUresult, cuMemcpyDtoDAsync, CUdeviceptr, dstDevice, CUdeviceptr, srcDevice, size_t, ByteCount, CUstream, hStream)
DEF_FN(CUresult, cuMemcpyDtoDAsync_v2_ptsz, CUdeviceptr, dstDevice, CUdeviceptr, srcDevice, size_t, ByteCount, CUstream, hStream)
DEF_FN(CUresult, cuMemcpy2DAsync, const CUDA_MEMCPY2D*, pCopy, CUstream, hStream)
DEF_FN(CUresult, cuMemcpy2DAsync_v2_ptsz, const CUDA_MEMCPY2D*, pCopy, CUstream, hStream)
DEF_FN(CUresult, cuMemcpy3DAsync, const CUDA_MEMCPY3D*, pCopy, CUstream, hStream)
DEF_FN(CUresult, cuMemcpy3DAsync_v2_ptsz, const CUDA_MEMCPY3D*, pCopy, CUstream, hStream)
DEF_FN(CUresult, cuMemcpyAsync, CUdeviceptr, dst, CUdeviceptr, src, size_t, ByteCount, CUstream, hStream)
DEF_FN(CUresult, cuMemcpyAsync_ptsz, CUdeviceptr, dst, CUdeviceptr, src, size_t, ByteCount, CUstream, hStream)
DEF_FN(CUresult, cuMemcpyPeer, CUdeviceptr, dstDevice, CUcontext, dstContext, CUdeviceptr, srcDevice, CUcontext, srcContext, size_t, ByteCount)
DEF_FN(CUresult, cuMemcpyPeer_ptds, CUdeviceptr, dstDevice, CUcontext, dstContext, CUdeviceptr, srcDevice, CUcontext, srcContext, size_t, ByteCount)
DEF_FN(CUresult, cuMemcpy3DPeer, const CUDA_MEMCPY3D_PEER*, pCopy)
DEF_FN(CUresult, cuMemcpy3DPeer_ptds, const CUDA_MEMCPY3D_PEER*, pCopy)
DEF_FN(CUresult, cuMemsetD8Async, CUdeviceptr, dstDevice, unsigned char, uc, size_t, N, CUstream, hStream)
DEF_FN(CUresult, cuMemsetD8Async_ptsz, CUdeviceptr, dstDevice, unsigned char, uc, size_t, N, CUstream, hStream)
DEF_FN(CUresult, cuMemsetD2D8Async, CUdeviceptr, dstDevice, size_t, dstPitch, unsigned char, uc, size_t, Width, size_t, Height, CUstream, hStream)
DEF_FN(CUresult, cuMemsetD2D8Async_ptsz, CUdeviceptr, dstDevice, size_t, dstPitch, unsigned char, uc, size_t, Width, size_t, Height, CUstream, hStream)
DEF_FN(CUresult, cuFuncSetCacheConfig, CUfunction, hfunc, CUfunc_cache, config)
DEF_FN(CUresult, cuFuncSetSharedMemConfig, CUfunction, hfunc, CUsharedconfig, config)
DEF_FN(CUresult, cuFuncGetAttribute, int*, pi, CUfunction_attribute, attrib, CUfunction, hfunc)
DEF_FN(CUresult, cuFuncSetAttribute, CUfunction, hfunc, CUfunction_attribute, attrib, int, value)
DEF_FN(CUresult, cuArrayDestroy, CUarray, hArray)
DEF_FN(CUresult, cuMipmappedArrayCreate, CUmipmappedArray*, pHandle, const CUDA_ARRAY3D_DESCRIPTOR*, pMipmappedArrayDesc, unsigned int, numMipmapLevels)
DEF_FN(CUresult, cuMipmappedArrayGetLevel, CUarray*, pLevelArray, CUmipmappedArray, hMipmappedArray, unsigned int, level)
DEF_FN(CUresult, cuMipmappedArrayDestroy, CUmipmappedArray, hMipmappedArray)
DEF_FN(CUresult, cuTexRefCreate, CUtexref*, pTexRef)
DEF_FN(CUresult, cuTexRefDestroy, CUtexref, hTexRef)
DEF_FN(CUresult, cuTexRefSetArray, CUtexref, hTexRef, CUarray, hArray, unsigned int, Flags)
DEF_FN(CUresult, cuTexRefSetMipmappedArray, CUtexref, hTexRef, CUmipmappedArray, hMipmappedArray, unsigned int, Flags)
DEF_FN(CUresult, cuTexRefSetFormat, CUtexref, hTexRef, CUarray_format, fmt, int, NumPackedComponents)
DEF_FN(CUresult, cuTexRefSetAddressMode, CUtexref, hTexRef, int, dim, CUaddress_mode, am)
DEF_FN(CUresult, cuTexRefSetFilterMode, CUtexref, hTexRef, CUfilter_mode, fm)
DEF_FN(CUresult, cuTexRefSetMipmapFilterMode, CUtexref, hTexRef, CUfilter_mode, fm)
DEF_FN(CUresult, cuTexRefSetMipmapLevelBias, CUtexref, hTexRef, float, bias)
DEF_FN(CUresult, cuTexRefSetMipmapLevelClamp, CUtexref, hTexRef, float, minMipmapLevelClamp, float, maxMipmapLevelClamp)
DEF_FN(CUresult, cuTexRefSetMaxAnisotropy, CUtexref, hTexRef, unsigned int, maxAniso)
DEF_FN(CUresult, cuTexRefSetFlags, CUtexref, hTexRef, unsigned int, Flags)
DEF_FN(CUresult, cuTexRefSetBorderColor, CUtexref, hTexRef, float*, pBorderColor)
DEF_FN(CUresult, cuTexRefGetBorderColor, float*, pBorderColor, CUtexref, hTexRef)
DEF_FN(CUresult, cuSurfRefSetArray, CUsurfref, hSurfRef, CUarray, hArray, unsigned int, Flags)
DEF_FN(CUresult, cuTexObjectCreate, CUtexObject*, pTexObject, const CUDA_RESOURCE_DESC*, pResDesc, const CUDA_TEXTURE_DESC*, pTexDesc, const CUDA_RESOURCE_VIEW_DESC*, pResViewDesc)
DEF_FN(CUresult, cuTexObjectDestroy, CUtexObject, texObject)
DEF_FN(CUresult, cuTexObjectGetResourceDesc, CUDA_RESOURCE_DESC*, pResDesc, CUtexObject, texObject)
DEF_FN(CUresult, cuTexObjectGetTextureDesc, CUDA_TEXTURE_DESC*, pTexDesc, CUtexObject, texObject)
DEF_FN(CUresult, cuTexObjectGetResourceViewDesc, CUDA_RESOURCE_VIEW_DESC*, pResViewDesc, CUtexObject, texObject)
DEF_FN(CUresult, cuSurfObjectCreate, CUsurfObject*, pSurfObject, const CUDA_RESOURCE_DESC*, pResDesc)
DEF_FN(CUresult, cuSurfObjectDestroy, CUsurfObject, surfObject)
DEF_FN(CUresult, cuSurfObjectGetResourceDesc, CUDA_RESOURCE_DESC*, pResDesc, CUsurfObject, surfObject)
DEF_FN(CUresult, cuImportExternalMemory, CUexternalMemory*, extMem_out, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC*, memHandleDesc)
DEF_FN(CUresult, cuExternalMemoryGetMappedBuffer, CUdeviceptr*, devPtr, CUexternalMemory, extMem, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC*, bufferDesc)
DEF_FN(CUresult, cuExternalMemoryGetMappedMipmappedArray, CUmipmappedArray*, mipmap, CUexternalMemory, extMem, const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC*, mipmapDesc)
DEF_FN(CUresult, cuDestroyExternalMemory, CUexternalMemory, extMem)
DEF_FN(CUresult, cuImportExternalSemaphore, CUexternalSemaphore*, extSem_out, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC*, semHandleDesc)
DEF_FN(CUresult, cuSignalExternalSemaphoresAsync, const CUexternalSemaphore*, extSemArray, const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS*, paramsArray, unsigned int, numExtSems, CUstream, stream)
DEF_FN(CUresult, cuSignalExternalSemaphoresAsync_ptsz, const CUexternalSemaphore*, extSemArray, const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS*, paramsArray, unsigned int, numExtSems, CUstream, stream)
DEF_FN(CUresult, cuWaitExternalSemaphoresAsync, const CUexternalSemaphore*, extSemArray, const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS*, paramsArray, unsigned int, numExtSems, CUstream, stream)
DEF_FN(CUresult, cuWaitExternalSemaphoresAsync_ptsz, const CUexternalSemaphore*, extSemArray, const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS*, paramsArray, unsigned int, numExtSems, CUstream, stream)
DEF_FN(CUresult, cuDestroyExternalSemaphore, CUexternalSemaphore, extSem)
#undef cuLaunchKernel
DEF_FN(CUresult, cuLaunchKernel, CUfunction, f, unsigned int, gridDimX, unsigned int, gridDimY, unsigned int, gridDimZ, unsigned int, blockDimX, unsigned int, blockDimY, unsigned int, blockDimZ, unsigned int, sharedMemBytes, CUstream, hStream, void**, kernelParams, void**, extra)
DEF_FN(CUresult, cuLaunchKernel_ptsz, CUfunction, f, unsigned int, gridDimX, unsigned int, gridDimY, unsigned int, gridDimZ, unsigned int, blockDimX, unsigned int, blockDimY, unsigned int, blockDimZ, unsigned int, sharedMemBytes, CUstream, hStream, void**, kernelParams, void**, extra)
#undef cuLaunchCooperativeKernel
DEF_FN(CUresult, cuLaunchCooperativeKernel, CUfunction, f, unsigned int, gridDimX, unsigned int, gridDimY, unsigned int, gridDimZ, unsigned int, blockDimX, unsigned int, blockDimY, unsigned int, blockDimZ, unsigned int, sharedMemBytes, CUstream, hStream, void**, kernelParams)
DEF_FN(CUresult, cuLaunchCooperativeKernel_ptsz, CUfunction, f, unsigned int, gridDimX, unsigned int, gridDimY, unsigned int, gridDimZ, unsigned int, blockDimX, unsigned int, blockDimY, unsigned int, blockDimZ, unsigned int, sharedMemBytes, CUstream, hStream, void**, kernelParams)
DEF_FN(CUresult, cuLaunchCooperativeKernelMultiDevice, CUDA_LAUNCH_PARAMS*, launchParamsList, unsigned int, numDevices, unsigned int, flags)
DEF_FN(CUresult, cuLaunchHostFunc, CUstream, hStream, CUhostFn, fn, void*, userData)
DEF_FN(CUresult, cuLaunchHostFunc_ptsz, CUstream, hStream, CUhostFn, fn, void*, userData)
DEF_FN(CUresult, cuEventCreate, CUevent*, phEvent, unsigned int, Flags)
DEF_FN(CUresult, cuEventRecord, CUevent, hEvent, CUstream, hStream)
DEF_FN(CUresult, cuEventRecord_ptsz, CUevent, hEvent, CUstream, hStream)
DEF_FN(CUresult, cuEventQuery, CUevent, hEvent)
DEF_FN(CUresult, cuEventSynchronize, CUevent, hEvent)
DEF_FN(CUresult, cuEventElapsedTime, float*, pMilliseconds, CUevent, hStart, CUevent, hEnd)
DEF_FN(CUresult, cuStreamWaitValue32, CUstream, stream, CUdeviceptr, addr, cuuint32_t, value, unsigned int, flags)
DEF_FN(CUresult, cuStreamWaitValue32_ptsz, CUstream, stream, CUdeviceptr, addr, cuuint32_t, value, unsigned int, flags)
DEF_FN(CUresult, cuStreamWriteValue32, CUstream, stream, CUdeviceptr, addr, cuuint32_t, value, unsigned int, flags)
DEF_FN(CUresult, cuStreamWriteValue32_ptsz, CUstream, stream, CUdeviceptr, addr, cuuint32_t, value, unsigned int, flags)
DEF_FN(CUresult, cuStreamWaitValue64, CUstream, stream, CUdeviceptr, addr, cuuint64_t, value, unsigned int, flags)
DEF_FN(CUresult, cuStreamWaitValue64_ptsz, CUstream, stream, CUdeviceptr, addr, cuuint64_t, value, unsigned int, flags)
DEF_FN(CUresult, cuStreamWriteValue64, CUstream, stream, CUdeviceptr, addr, cuuint64_t, value, unsigned int, flags)
DEF_FN(CUresult, cuStreamWriteValue64_ptsz, CUstream, stream, CUdeviceptr, addr, cuuint64_t, value, unsigned int, flags)
DEF_FN(CUresult, cuStreamBatchMemOp, CUstream, stream, unsigned int, count, CUstreamBatchMemOpParams*, paramArray, unsigned int, flags)
DEF_FN(CUresult, cuStreamBatchMemOp_ptsz, CUstream, stream, unsigned int, count, CUstreamBatchMemOpParams*, paramArray, unsigned int, flags)
DEF_FN(CUresult, cuStreamCreate, CUstream*, phStream, unsigned int, Flags)
DEF_FN(CUresult, cuStreamCreateWithPriority, CUstream*, phStream, unsigned int, flags, int, priority)
DEF_FN(CUresult, cuStreamGetPriority, CUstream, hStream, int*, priority)
DEF_FN(CUresult, cuStreamGetPriority_ptsz, CUstream, hStream, int*, priority)
DEF_FN(CUresult, cuStreamGetFlags, CUstream, hStream, unsigned int*, flags)
DEF_FN(CUresult, cuStreamGetFlags_ptsz, CUstream, hStream, unsigned int*, flags)
DEF_FN(CUresult, cuStreamGetCtx, CUstream, hStream, CUcontext*, pctx)
DEF_FN(CUresult, cuStreamWaitEvent, CUstream, hStream, CUevent, hEvent, unsigned int, Flags)
DEF_FN(CUresult, cuStreamWaitEvent_ptsz, CUstream, hStream, CUevent, hEvent, unsigned int, Flags)
DEF_FN(CUresult, cuStreamAddCallback, CUstream, hStream, CUstreamCallback, callback, void*, userData, unsigned int, flags)
DEF_FN(CUresult, cuStreamAddCallback_ptsz, CUstream, hStream, CUstreamCallback, callback, void*, userData, unsigned int, flags)
DEF_FN(CUresult, cuStreamSynchronize, CUstream, hStream)
DEF_FN(CUresult, cuStreamSynchronize_ptsz, CUstream, hStream)
DEF_FN(CUresult, cuStreamQuery, CUstream, hStream)
DEF_FN(CUresult, cuStreamQuery_ptsz, CUstream, hStream)
DEF_FN(CUresult, cuStreamAttachMemAsync, CUstream, hStream, CUdeviceptr, dptr, size_t, length, unsigned int, flags)
DEF_FN(CUresult, cuStreamAttachMemAsync_ptsz, CUstream, hStream, CUdeviceptr, dptr, size_t, length, unsigned int, flags)
DEF_FN(CUresult, cuDeviceCanAccessPeer, int*, canAccessPeer, CUdevice, dev, CUdevice, peerDev)
DEF_FN(CUresult, cuCtxEnablePeerAccess, CUcontext, peerContext, unsigned int, Flags)
DEF_FN(CUresult, cuCtxDisablePeerAccess, CUcontext, peerContext)
DEF_FN(CUresult, cuIpcGetEventHandle, CUipcEventHandle*, pHandle, CUevent, event)
DEF_FN(CUresult, cuIpcOpenEventHandle, CUevent*, phEvent, CUipcEventHandle, handle)
DEF_FN(CUresult, cuIpcGetMemHandle, CUipcMemHandle*, pHandle, CUdeviceptr, dptr)
DEF_FN(CUresult, cuIpcOpenMemHandle, CUdeviceptr*, pdptr, CUipcMemHandle, handle, unsigned int, Flags)
DEF_FN(CUresult, cuIpcCloseMemHandle, CUdeviceptr, dptr)
DEF_FN(CUresult, cuGraphicsUnregisterResource, CUgraphicsResource, resource)
DEF_FN(CUresult, cuGraphicsMapResources, unsigned int, count, CUgraphicsResource*, resources, CUstream, hStream)
DEF_FN(CUresult, cuGraphicsMapResources_ptsz, unsigned int, count, CUgraphicsResource*, resources, CUstream, hStream)
DEF_FN(CUresult, cuGraphicsUnmapResources, unsigned int, count, CUgraphicsResource*, resources, CUstream, hStream)
DEF_FN(CUresult, cuGraphicsUnmapResources_ptsz, unsigned int, count, CUgraphicsResource*, resources, CUstream, hStream)
DEF_FN(CUresult, cuGraphicsSubResourceGetMappedArray, CUarray*, pArray, CUgraphicsResource, resource, unsigned int, arrayIndex, unsigned int, mipLevel)
DEF_FN(CUresult, cuGraphicsResourceGetMappedMipmappedArray, CUmipmappedArray*, pMipmappedArray, CUgraphicsResource, resource)
DEF_FN(CUresult, cuGraphicsResourceGetMappedPointer, CUdeviceptr*, pDevPtr, size_t*, pSize, CUgraphicsResource, resource)
DEF_FN(CUresult, cuGraphicsResourceSetMapFlags, CUgraphicsResource, resource, unsigned int, flags)
//DEF_FN(CUresult, cuGetExportTable, const void**, ppExportTable, const CUuuid*, pExportTableId)

// This function returns an array of 8 function pointers to hidden functions inside libcuda.so
CUresult cuGetExportTable(const void** ppExportTable, const CUuuid* pExportTableId)
{
	enum clnt_stat retval;
    rpc_uuid uuid;
    int_result result;
    const void* p1_data = *ppExportTable;
    char idstr[64];
    uint64_t tablesize = 0;
    if (pExportTableId == NULL) {
        return CUDA_ERROR_UNKNOWN;
    }
    for (int i=0; i < 16;++i) {
        sprintf(idstr+i*3, "%02x ", pExportTableId->bytes[i] & 0xFF);
    }
    uuid.rpc_uuid_val = malloc(16);
    uuid.rpc_uuid_len = 16;
    memcpy(uuid.rpc_uuid_val, pExportTableId->bytes, 16);
    //printf("precall %p->%p\n", ppExportTable, *ppExportTable);
    retval = rpc_cugetexporttable_1(uuid, &result, clnt);
    //printf("postcall %p->%p\n", p1_data, *((void**)p1_data));
    tablesize = result.int_result_u.data;
    printf("[rpc] %s(%p, %s) = %d, result = %d\n", __FUNCTION__, p1_data, idstr, result.err,
                                                   tablesize);

	if (retval != RPC_SUCCESS) {
		fprintf(stderr, "[rpc] %s failed.", __FUNCTION__);
        return CUDA_ERROR_UNKNOWN;
	}
    *ppExportTable = cd_client_hidden_get();

    cd_client_hidden_incr();
/*
    memcpy(*((void**)ppExportTable), p1_data, tablesize*sizeof(void*));

    if (((uint32_t*)p1_data)[1] > 0) {
        for (int i=0; i < tablesize; ++i) {
            ((void**)*ppExportTable)[i] =
                cd_client_hidden_replace(((void**)*ppExportTable)[i], i);
        }
    } else {
        for (int i=1; i < tablesize; ++i) {
            ((void**)*ppExportTable)[i] =
                cd_client_hidden_replace(((void**)*ppExportTable)[i], i-1);
        }
    }
*/
    /*if (test2_1fptr == NULL) {
        test2_1fptr = ((void**)*ppExportTable)[2];
        ((void**)*ppExportTable)[2] = test2_1;
    } else if (test2_2fptr == NULL) {
        test2_2fptr = ((void**)*ppExportTable)[2];
        ((void**)*ppExportTable)[2] = test2_2;
    } else {
        printf("\tthis is too much\n");
        ((void**)*ppExportTable)[2] = NULL;
    }*/
    //((void**)*ppExportTable)[1] = test1;
    //((void**)*ppExportTable)[3] = test1;
    for (int i=0; i < tablesize; ++i)
        printf("\t%p\n", ((void**)*ppExportTable)[i]);
    return result.err;
}
DEF_FN(CUresult, cuOccupancyMaxActiveBlocksPerMultiprocessor, int*, numBlocks, CUfunction, func, int, blockSize, size_t, dynamicSMemSize)
DEF_FN(CUresult, cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags, int*, numBlocks, CUfunction, func, int, blockSize, size_t, dynamicSMemSize, unsigned int, flags)
DEF_FN(CUresult, cuMemAdvise, CUdeviceptr, devPtr, size_t, count, CUmem_advise, advice, CUdevice, device)
DEF_FN(CUresult, cuMemPrefetchAsync, CUdeviceptr, devPtr, size_t, count, CUdevice, dstDevice, CUstream, hStream)
DEF_FN(CUresult, cuMemPrefetchAsync_ptsz, CUdeviceptr, devPtr, size_t, count, CUdevice, dstDevice, CUstream, hStream)
DEF_FN(CUresult, cuMemRangeGetAttribute, void*, data, size_t, dataSize, CUmem_range_attribute, attribute, CUdeviceptr, devPtr, size_t, count)
DEF_FN(CUresult, cuMemRangeGetAttributes, void**, data, size_t*, dataSizes, CUmem_range_attribute*, attributes, size_t, numAttributes, CUdeviceptr, devPtr, size_t, count)
DEF_FN(CUresult, cuGetErrorString, CUresult, error, const char**, pStr)
DEF_FN(CUresult, cuGetErrorName, CUresult, error, const char**, pStr)
DEF_FN(CUresult, cuGraphCreate, CUgraph*, phGraph, unsigned int, flags)
DEF_FN(CUresult, cuGraphAddKernelNode, CUgraphNode*, phGraphNode, CUgraph, hGraph, const CUgraphNode*, dependencies, size_t, numDependencies, const CUDA_KERNEL_NODE_PARAMS*, nodeParams)
DEF_FN(CUresult, cuGraphKernelNodeGetParams, CUgraphNode, hNode, CUDA_KERNEL_NODE_PARAMS*, nodeParams)
DEF_FN(CUresult, cuGraphKernelNodeSetParams, CUgraphNode, hNode, const CUDA_KERNEL_NODE_PARAMS*, nodeParams)
DEF_FN(CUresult, cuGraphAddMemcpyNode, CUgraphNode*, phGraphNode, CUgraph, hGraph, const CUgraphNode*, dependencies, size_t, numDependencies, const CUDA_MEMCPY3D*, copyParams, CUcontext, ctx)
DEF_FN(CUresult, cuGraphMemcpyNodeGetParams, CUgraphNode, hNode, CUDA_MEMCPY3D*, nodeParams)
DEF_FN(CUresult, cuGraphMemcpyNodeSetParams, CUgraphNode, hNode, const CUDA_MEMCPY3D*, nodeParams)
DEF_FN(CUresult, cuGraphAddMemsetNode, CUgraphNode*, phGraphNode, CUgraph, hGraph, const CUgraphNode*, dependencies, size_t, numDependencies, const CUDA_MEMSET_NODE_PARAMS*, memsetParams, CUcontext, ctx)
DEF_FN(CUresult, cuGraphMemsetNodeGetParams, CUgraphNode, hNode, CUDA_MEMSET_NODE_PARAMS*, nodeParams)
DEF_FN(CUresult, cuGraphMemsetNodeSetParams, CUgraphNode, hNode, const CUDA_MEMSET_NODE_PARAMS*, nodeParams)
DEF_FN(CUresult, cuGraphAddHostNode, CUgraphNode*, phGraphNode, CUgraph, hGraph, const CUgraphNode*, dependencies, size_t, numDependencies, const CUDA_HOST_NODE_PARAMS*, nodeParams)
DEF_FN(CUresult, cuGraphHostNodeGetParams, CUgraphNode, hNode, CUDA_HOST_NODE_PARAMS*, nodeParams)
DEF_FN(CUresult, cuGraphHostNodeSetParams, CUgraphNode, hNode, const CUDA_HOST_NODE_PARAMS*, nodeParams)
DEF_FN(CUresult, cuGraphAddChildGraphNode, CUgraphNode*, phGraphNode, CUgraph, hGraph, const CUgraphNode*, dependencies, size_t, numDependencies, CUgraph, childGraph)
DEF_FN(CUresult, cuGraphChildGraphNodeGetGraph, CUgraphNode, hNode, CUgraph*, phGraph)
DEF_FN(CUresult, cuGraphAddEmptyNode, CUgraphNode*, phGraphNode, CUgraph, hGraph, const CUgraphNode*, dependencies, size_t, numDependencies)
DEF_FN(CUresult, cuGraphClone, CUgraph*, phGraphClone, CUgraph, originalGraph)
DEF_FN(CUresult, cuGraphNodeFindInClone, CUgraphNode*, phNode, CUgraphNode, hOriginalNode, CUgraph, hClonedGraph)
DEF_FN(CUresult, cuGraphNodeGetType, CUgraphNode, hNode, CUgraphNodeType*, type)
DEF_FN(CUresult, cuGraphGetNodes, CUgraph, hGraph, CUgraphNode*, nodes, size_t*, numNodes)
DEF_FN(CUresult, cuGraphGetRootNodes, CUgraph, hGraph, CUgraphNode*, rootNodes, size_t*, numRootNodes)
DEF_FN(CUresult, cuGraphGetEdges, CUgraph, hGraph, CUgraphNode*, from, CUgraphNode*, to, size_t*, numEdges)
DEF_FN(CUresult, cuGraphNodeGetDependencies, CUgraphNode, hNode, CUgraphNode*, dependencies, size_t*, numDependencies)
DEF_FN(CUresult, cuGraphNodeGetDependentNodes, CUgraphNode, hNode, CUgraphNode*, dependentNodes, size_t*, numDependentNodes)
DEF_FN(CUresult, cuGraphAddDependencies, CUgraph, hGraph, const CUgraphNode*, from, const CUgraphNode*, to, size_t, numDependencies)
DEF_FN(CUresult, cuGraphRemoveDependencies, CUgraph, hGraph, const CUgraphNode*, from, const CUgraphNode*, to, size_t, numDependencies)
DEF_FN(CUresult, cuGraphInstantiate, CUgraphExec*, phGraphExec, CUgraph, hGraph, CUgraphNode*, phErrorNode, char*, logBuffer, size_t, bufferSize)
DEF_FN(CUresult, cuGraphLaunch, CUgraphExec, hGraphExec, CUstream, hStream)
DEF_FN(CUresult, cuGraphLaunch_ptsz, CUgraphExec, hGraphExec, CUstream, hStream)
DEF_FN(CUresult, cuGraphExecDestroy, CUgraphExec, hGraphExec)
DEF_FN(CUresult, cuGraphDestroyNode, CUgraphNode, hNode)
DEF_FN(CUresult, cuGraphDestroy, CUgraph, hGraph)
DEF_FN(CUresult, cuGraphDestroy_ptsz, CUgraph, hGraph)
DEF_FN(CUresult, cuStreamBeginCapture_ptsz, CUstream, hStream)
DEF_FN(CUresult, cuStreamBeginCapture, CUstream, hStream, CUstreamCaptureMode, mode)
#undef cuStreamBeginCapture
DEF_FN(CUresult, cuStreamBeginCapture, CUstream, hStream, CUstreamCaptureMode, mode)
DEF_FN(CUresult, cuStreamBeginCapture_v2_ptsz, CUstream, hStream)
DEF_FN(CUresult, cuStreamEndCapture, CUstream, hStream, CUgraph*, phGraph)
DEF_FN(CUresult, cuStreamEndCapture_ptsz, CUstream, hStream, CUgraph*, phGraph)
DEF_FN(CUresult, cuStreamIsCapturing, CUstream, hStream, CUstreamCaptureStatus*, captureStatus)
DEF_FN(CUresult, cuStreamIsCapturing_ptsz, CUstream, hStream, CUstreamCaptureStatus*, captureStatus)
DEF_FN(CUresult, cuThreadExchangeStreamCaptureMode, CUstreamCaptureMode*, mode)
DEF_FN(CUresult, cuStreamGetCaptureInfo, CUstream, hStream, CUstreamCaptureStatus*, captureStatus, cuuint64_t*, id)
DEF_FN(CUresult, cuStreamGetCaptureInfo_ptsz, CUstream, hStream, CUstreamCaptureStatus*, captureStatus, cuuint64_t*, id)
DEF_FN(CUresult, cuGraphExecKernelNodeSetParams, CUgraphExec, hGraphExec, CUgraphNode, hNode, const CUDA_KERNEL_NODE_PARAMS*, nodeParams)

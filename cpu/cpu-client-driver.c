#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cudaGL.h>
#include <cudaEGL.h>
#include <vdpau/vdpau.h>
#include <cudaVDPAU.h>
#include <elf.h>

#include <driver_types.h>
#include <string.h>

#include "cpu-libwrap.h"
#include "cpu-client-driver-hidden.h"
#include "cpu_rpc_prot.h"
#include "cpu-common.h"
#include "cpu-utils.h"
#include "cpu-elf2.h"


//DEF_FN(CUresult, cuProfilerInitialize, const char*, configFile, const char*, outputFile, CUoutput_mode, outputMode)
//DEF_FN(CUresult, cuProfilerStart)
//DEF_FN(CUresult, cuProfilerStop)
DEF_FN(CUresult, cuVDPAUGetDevice, CUdevice*, pDevice, VdpDevice, vdpDevice, VdpGetProcAddress*, vdpGetProcAddress)
#undef cuVDPAUCtxCreate
DEF_FN(CUresult, cuVDPAUCtxCreate, CUcontext*, pCtx, unsigned int, flags, CUdevice, device, VdpDevice, vdpDevice, VdpGetProcAddress*, vdpGetProcAddress)
DEF_FN(CUresult, cuGraphicsVDPAURegisterVideoSurface, CUgraphicsResource*, pCudaResource, VdpVideoSurface, vdpSurface, unsigned int, flags)
DEF_FN(CUresult, cuGraphicsVDPAURegisterOutputSurface, CUgraphicsResource*, pCudaResource, VdpOutputSurface, vdpSurface, unsigned int, flags)

#undef cuDeviceTotalMem
CUresult cuDeviceTotalMem(size_t* bytes, CUdevice dev)
{
	enum clnt_stat retval;
    u64_result result;
    retval = rpc_cudevicetotalmem_1(dev, &result, clnt);
    printf("[rpc] %s = %d, result %u\n", __FUNCTION__, result.err,
                                        result.u64_result_u.u64);
	if (retval != RPC_SUCCESS) {
		fprintf(stderr, "[rpc] %s failed.", __FUNCTION__);
        return CUDA_ERROR_UNKNOWN;
	}
    *bytes = result.u64_result_u.u64;
    return result.err;
}

#undef cuCtxCreate
CUresult cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev)
{
    DEF_FN_PTR(CUresult, CUcontext*, unsigned int, CUdevice);
    DEF_DLSYM(CUresult, cuCtxCreate)
    CAL_FN_PTR(pctx, flags, dev);
    printf("%s(%p, %u, %d) = %d\n", __FUNCTION__, pctx, flags, dev, ret);
    return ret;
}
DEF_FN(CUresult, cuCtxSynchronize)
#undef cuModuleGetGlobal
DEF_FN(CUresult, cuModuleGetGlobal, CUdeviceptr*, dptr, size_t*, bytes, CUmodule, hmod, const char*, name)
#undef cuMemGetInfo
DEF_FN(CUresult, cuMemGetInfo, size_t*, free, size_t*, total)

#undef cuMemAlloc
CUresult cuMemAlloc(CUdeviceptr* dptr, size_t bytesize)
{
	enum clnt_stat retval;
    ptr_result result;
    retval = rpc_cumemalloc_1(bytesize, &result, clnt);
    //printf("pre %s(%p->%p, %lu) = %d\n", __FUNCTION__, dptr, *dptr, bytesize, ret);
    printf("[rpc] %s(%lu) = %d, result %p\n", __FUNCTION__, bytesize, result.err, result.ptr_result_u.ptr);
	if (retval != RPC_SUCCESS) {
		fprintf(stderr, "[rpc] %s failed.", __FUNCTION__);
        return CUDA_ERROR_UNKNOWN;
	}
    *dptr = result.ptr_result_u.ptr;
    //printf("post %s(%p->%p, %lu) = %d\n", __FUNCTION__, dptr, *dptr, bytesize, ret);
    return result.err;
}

#undef cuMemAllocPitch
DEF_FN(CUresult, cuMemAllocPitch, CUdeviceptr*, dptr, size_t*, pPitch, size_t, WidthInBytes, size_t, Height, unsigned int, ElementSizeBytes)
#undef cuMemFree
DEF_FN(CUresult, cuMemFree, CUdeviceptr, dptr)
#undef cuMemGetAddressRange
DEF_FN(CUresult, cuMemGetAddressRange, CUdeviceptr*, pbase, size_t*, psize, CUdeviceptr, dptr)
#undef cuMemHostGetDevicePointer
DEF_FN(CUresult, cuMemHostGetDevicePointer, CUdeviceptr*, pdptr, void*, p, unsigned int, Flags)
#undef cuMemHostRegister
DEF_FN(CUresult, cuMemHostRegister, void*, p, size_t, bytesize, unsigned int, Flags)
#undef cuMemsetD8
DEF_FN(CUresult, cuMemsetD8, CUdeviceptr, dstDevice, unsigned char, uc, size_t, N);
DEF_FN(CUresult, cuMemsetD8_v2_ptds, CUdeviceptr, dstDevice, unsigned char, uc, size_t, N);
#undef cuMemsetD2D8
DEF_FN(CUresult, cuMemsetD2D8, CUdeviceptr, dstDevice, size_t, dstPitch, unsigned char, uc, size_t, Width, size_t, Height)
DEF_FN(CUresult, cuMemsetD2D8_v2_ptds, CUdeviceptr, dstDevice, size_t, dstPitch, unsigned char, uc, size_t, Width, size_t, Height)
#undef cuEventDestroy
DEF_FN(CUresult, cuEventDestroy, CUevent, hEvent)
#undef cuStreamDestroy
DEF_FN(CUresult, cuStreamDestroy, CUstream, hStream)
#undef cuGLCtxCreate
DEF_FN(CUresult, cuGLCtxCreate, CUcontext*, pCtx, unsigned int, Flags, CUdevice, device)
#undef cuArrayCreate
DEF_FN(CUresult, cuArrayCreate, CUarray*, pHandle, const CUDA_ARRAY_DESCRIPTOR*, pAllocateArray)
#undef cuArrayGetDescriptor
DEF_FN(CUresult, cuArrayGetDescriptor, CUDA_ARRAY_DESCRIPTOR*, pArrayDescriptor, CUarray, hArray)
#undef cuArray3DCreate
DEF_FN(CUresult, cuArray3DCreate, CUarray*, pHandle, const CUDA_ARRAY3D_DESCRIPTOR*, pAllocateArray)
#undef cuArray3DGetDescriptor
DEF_FN(CUresult, cuArray3DGetDescriptor, CUDA_ARRAY3D_DESCRIPTOR*, pArrayDescriptor, CUarray, hArray)
#undef cuTexRefSetAddress2D
DEF_FN(CUresult, cuTexRefSetAddress2D, CUtexref, hTexRef, const CUDA_ARRAY_DESCRIPTOR*, desc, CUdeviceptr, dptr, size_t, Pitch)
#undef cuTexRefSetAddress
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

//DEF_FN(CUresult, cuDeviceGetName, char*, name, int, len, CUdevice, dev)
CUresult cuDeviceGetName(char* name, int len, CUdevice dev)
{
	enum clnt_stat retval;
    str_result result;
    result.str_result_u.str = malloc(128);
    retval = rpc_cudevicegetname_1(dev, &result, clnt);
	if (retval != RPC_SUCCESS) {
		fprintf(stderr, "[rpc] %s failed.", __FUNCTION__);
        return CUDA_ERROR_UNKNOWN;
	}
    if (result.str_result_u.str == NULL) {
		fprintf(stderr, "[rpc] %s str is NULL.", __FUNCTION__);
        return CUDA_ERROR_UNKNOWN;
    }
    printf("[rpc] %s = %d, result \"%s\"\n", __FUNCTION__, result.err,
                                        result.str_result_u.str);
    strncpy(name, result.str_result_u.str, len);
    free(result.str_result_u.str);
    return result.err;
}

//DEF_FN(CUresult, cuDeviceGetUuid, CUuuid*, uuid, CUdevice, dev)

/* CUuuid = struct { char bytes[16] };
 * CUdevice = int
 */
CUresult cuDeviceGetUuid(CUuuid* uuid, CUdevice dev)
{
	enum clnt_stat retval;
    str_result result;
    retval = rpc_cudevicegetuuid_1(dev, &result, clnt);
    printf("[rpc] %s = %d, result (uuid)\n", __FUNCTION__, result.err);

	if (retval != RPC_SUCCESS) {
		fprintf(stderr, "[rpc] %s failed.", __FUNCTION__);
        return CUDA_ERROR_UNKNOWN;
	}
    memcpy(uuid->bytes, result.str_result_u.str, 16);
    return result.err;
}

DEF_FN(CUresult, cuDeviceGetLuid, char*, luid, unsigned int*, deviceNodeMask, CUdevice, dev)

CUresult cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev)
{
	enum clnt_stat retval;
    int_result result;
    retval = rpc_cudevicegetattribute_1(attrib, dev, &result, clnt);
    printf("[rpc] %s = %d, result %d\n", __FUNCTION__, result.err,
                                        result.int_result_u.data);
	if (retval != RPC_SUCCESS) {
		fprintf(stderr, "[rpc] %s failed.", __FUNCTION__);
        return CUDA_ERROR_UNKNOWN;
	}
    *pi = result.int_result_u.data;
    return result.err;
}

CUresult cuDeviceGetProperties(CUdevprop* prop, CUdevice dev)
{
	enum clnt_stat retval;
    mem_result result;
    if (prop == NULL) {
        LOGE(LOG_ERROR, "%s: prop is NULL", __FUNCTION__);
        return CUDA_ERROR_INVALID_VALUE;
    }
    retval = rpc_cudevicegetproperties_1(dev, &result, clnt);
    LOGE(LOG_DEBUG, "%s = %d, result len: %d", __FUNCTION__, result.err,
                                        result.mem_result_u.data.mem_data_len);
	if (retval != RPC_SUCCESS) {
		fprintf(stderr, "[rpc] %s failed.", __FUNCTION__);
        return CUDA_ERROR_UNKNOWN;
	}
    if (result.mem_result_u.data.mem_data_len != sizeof(CUdevprop)) {
        LOGE(LOG_ERROR, "%s: size mismatch", __FUNCTION__);
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (memcpy(prop, result.mem_result_u.data.mem_data_val, sizeof(CUdevprop)) == NULL) {
        LOGE(LOG_ERROR, "%s: memcpy failed", __FUNCTION__);
        return CUDA_ERROR_UNKNOWN;
    }
    return result.err;
}
CUresult cuDeviceComputeCapability(int* major, int* minor, CUdevice dev)
{
    enum clnt_stat retval;
    dint_result result;
    if (major == NULL || minor == NULL) {
        LOGE(LOG_ERROR, "%s: major or minor is NULL", __FUNCTION__);
        return CUDA_ERROR_INVALID_VALUE;
    }
    retval = rpc_cudevicecomputecapability_1(dev, &result, clnt);
    LOGE(LOG_DEBUG, "%s = %d, result %d, %d", __FUNCTION__, result.err,
                                        result.dint_result_u.data.i1,
                                        result.dint_result_u.data.i2);
    if (retval != RPC_SUCCESS) {
        fprintf(stderr, "[rpc] %s failed.", __FUNCTION__);
        return CUDA_ERROR_UNKNOWN;
    }
    *major = result.dint_result_u.data.i1;
    *minor = result.dint_result_u.data.i2;
    return result.err;
} 

DEF_FN(CUresult, cuDeviceGetByPCIBusId, CUdevice*, dev, const char*, pciBusId)
CUresult cuDeviceGetP2PAttribute ( int* value, CUdevice_P2PAttribute attrib, CUdevice srcDevice, CUdevice dstDevice ) 
{
	enum clnt_stat retval;
    int_result result;
    retval = rpc_cudevicegetp2pattribute_1((int)attrib, (ptr)srcDevice, (ptr)dstDevice, &result, clnt);
    LOGE(LOG_DEBUG, "[rpc] %s(%d, %p, %p) = %d, result %s", __FUNCTION__, attrib, srcDevice, dstDevice, result.err, result.int_result_u.data);
	if (retval != RPC_SUCCESS) {
		fprintf(stderr, "[rpc] %s failed.", __FUNCTION__);
        return CUDA_ERROR_UNKNOWN;
	}
    return result.err;
}

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
	enum clnt_stat retval;
    ptr_result result;
    retval = rpc_cudeviceprimaryctxretain_1(dev, &result, clnt);
    printf("[rpc] %s = %d, result %p\n", __FUNCTION__, result.err,
                                        result.ptr_result_u.ptr);
	if (retval != RPC_SUCCESS) {
		fprintf(stderr, "[rpc] %s failed.", __FUNCTION__);
        return CUDA_ERROR_UNKNOWN;
	}
    *pctx = (CUcontext)result.ptr_result_u.ptr;
    return result.err;
}
#undef cuDevicePrimaryCtxRelease
DEF_FN(CUresult, cuDevicePrimaryCtxRelease, CUdevice, dev)
#undef cuDevicePrimaryCtxSetFlags
DEF_FN(CUresult, cuDevicePrimaryCtxSetFlags, CUdevice, dev, unsigned int, flags)
CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int* flags, int* active)
{
	enum clnt_stat retval;
    dint_result result;
    if (flags == NULL || active == NULL) {
        LOGE(LOG_ERROR, "%s flags or active is NULL.", __FUNCTION__);
        return CUDA_ERROR_INVALID_VALUE;
    }
    retval = rpc_cudeviceprimaryctxgetstate_1(dev, &result, clnt);
    LOGE(LOG_DEBUG, "%s = %d, result %d %d", __FUNCTION__, result.err,
                                        result.dint_result_u.data.i1,
                                        result.dint_result_u.data.i2);
	if (retval != RPC_SUCCESS) {
		LOGE(LOG_ERROR, "%s failed.", __FUNCTION__);
        return CUDA_ERROR_UNKNOWN;
	}
    *flags = result.dint_result_u.data.i1;
    *active = result.dint_result_u.data.i2; 
    return result.err;
}
#undef cuDevicePrimaryCtxReset
DEF_FN(CUresult, cuDevicePrimaryCtxReset, CUdevice, dev)
DEF_FN(CUresult, cuCtxGetFlags, unsigned int*, flags)
//DEF_FN(CUresult, cuCtxSetCurrent, CUcontext, ctx)
CUresult cuCtxSetCurrent(CUcontext ctx)
{
	enum clnt_stat retval;
    int result;
    retval = rpc_cuctxsetcurrent_1((uint64_t)ctx, &result, clnt);
    printf("[rpc] %s = %d, result %d\n", __FUNCTION__, result);
	if (retval != RPC_SUCCESS) {
		fprintf(stderr, "[rpc] %s failed.", __FUNCTION__);
        return CUDA_ERROR_UNKNOWN;
	}
    return result;
}
//DEF_FN(CUresult, cuCtxGetCurrent, CUcontext*, pctx)
CUresult cuCtxGetCurrent(CUcontext *pctx)
{
	enum clnt_stat retval;
    ptr_result result;
    retval = rpc_cuctxgetcurrent_1(&result, clnt);
    printf("[rpc] %s(%p) = %d, result %p\n", __FUNCTION__, pctx, result.err,
                                        result.ptr_result_u.ptr);
	if (retval != RPC_SUCCESS) {
		fprintf(stderr, "[rpc] %s failed.", __FUNCTION__);
        return CUDA_ERROR_UNKNOWN;
	}
    *pctx = (CUcontext)result.ptr_result_u.ptr;
    return result.err;
}
DEF_FN(CUresult, cuCtxDetach, CUcontext, ctx)
DEF_FN(CUresult, cuCtxGetApi2Version, CUcontext, ctx, unsigned int*, version)
//DEF_FN(CUresult, cuCtxGetDevice, CUdevice*, device)
CUresult cuCtxGetDevice(CUdevice *device)
{
	enum clnt_stat retval;
    int_result result;
    retval = rpc_cuctxgetdevice_1(&result, clnt);
    printf("[rpc] %s = %d, result %d\n", __FUNCTION__, result.err,
                                        result.int_result_u.data);
	if (retval != RPC_SUCCESS) {
		fprintf(stderr, "[rpc] %s failed.", __FUNCTION__);
        return CUDA_ERROR_UNKNOWN;
	}
    *device = (CUdevice)result.int_result_u.data;
    return result.err;
}
DEF_FN(CUresult, cuCtxGetLimit, size_t*, pvalue, CUlimit, limit)
DEF_FN(CUresult, cuCtxSetLimit, CUlimit, limit, size_t, value)
DEF_FN(CUresult, cuCtxGetCacheConfig, CUfunc_cache*, pconfig)
DEF_FN(CUresult, cuCtxSetCacheConfig, CUfunc_cache, config)
DEF_FN(CUresult, cuCtxGetSharedMemConfig, CUsharedconfig*, pConfig)
DEF_FN(CUresult, cuCtxGetStreamPriorityRange, int*, leastPriority, int*, greatestPriority)
DEF_FN(CUresult, cuCtxSetSharedMemConfig, CUsharedconfig, config)
DEF_FN(CUresult, cuCtxSynchronize, void)
CUresult cuModuleLoad(CUmodule* module, const char* fname)
{
	enum clnt_stat retval;
    ptr_result result;

    if (fname == NULL) {
        LOGE(LOG_ERROR, "fname is NULL!");
        return CUDA_ERROR_FILE_NOT_FOUND;
    }
    if (cpu_utils_parameter_info(&kernel_infos, (char*)fname) != 0) {
        LOGE(LOG_ERROR, "could not get kernel infos from %s", fname);
        return CUDA_ERROR_FILE_NOT_FOUND;
    }

    retval = rpc_cumoduleload_1((char*)fname, &result, clnt);
    printf("[rpc] %s(%s) = %d, result %p\n", __FUNCTION__, fname, result.err, (void*)result.ptr_result_u.ptr);
	if (retval != RPC_SUCCESS) {
		fprintf(stderr, "[rpc] %s failed.", __FUNCTION__);
        return CUDA_ERROR_UNKNOWN;
	}
    if (module != NULL) {
       *module = (CUmodule)result.ptr_result_u.ptr;
    }
    return result.err;
}


CUresult cuModuleLoadData(CUmodule* module, const void* image)
{
	enum clnt_stat retval;
    ptr_result result;
    mem_data mem;

    if (image == NULL) {
        LOGE(LOG_ERROR, "image is NULL!");
        return CUDA_ERROR_INVALID_IMAGE;
    }
    Elf64_Ehdr *ehdr = (Elf64_Ehdr*)image;

    if (ehdr->e_ident[EI_MAG0] != ELFMAG0 ||
        ehdr->e_ident[EI_MAG1] != ELFMAG1 ||
        ehdr->e_ident[EI_MAG2] != ELFMAG2 ||
        ehdr->e_ident[EI_MAG3] != ELFMAG3) {
        LOGE(LOG_ERROR, "image is not an ELF!");
        return CUDA_ERROR_INVALID_IMAGE;
    }

    mem.mem_data_len = ehdr->e_shoff + ehdr->e_shnum * ehdr->e_shentsize;
    mem.mem_data_val = (uint8_t*)image;

    LOGE(LOG_DEBUG, "image_size = %#0zx", mem.mem_data_len);
    
    if (elf2_parameter_info(&kernel_infos, mem.mem_data_val, mem.mem_data_len) != 0) {
        LOGE(LOG_ERROR, "could not get kernel infos from memory");
        return CUDA_ERROR_INVALID_IMAGE;
    }

    retval = rpc_cumoduleloaddata_1(mem, &result, clnt);
    printf("[rpc] %s(%p) = %d, result %p\n", __FUNCTION__, image, result.err, (void*)result.ptr_result_u.ptr);
	if (retval != RPC_SUCCESS) {
		fprintf(stderr, "[rpc] %s failed.", __FUNCTION__);
        return CUDA_ERROR_UNKNOWN;
	}
    if (module != NULL) {
       *module = (CUmodule)result.ptr_result_u.ptr;
    }
    return result.err;
}

DEF_FN(CUresult, cuModuleLoadDataEx, CUmodule*, module, const void*, image, unsigned int, numOptions, CUjit_option*, options, void**, optionValues)
DEF_FN(CUresult, cuModuleLoadFatBinary, CUmodule*, module, const void*, fatCubin)
CUresult cuModuleUnload(CUmodule hmod)
{
	enum clnt_stat retval;
    int result;

    retval = rpc_cumoduleunload_1((ptr)hmod, &result, clnt);
    LOGE(LOG_DEBUG, "[rpc] %s(%p) = %d", __FUNCTION__, hmod, result);
	if (retval != RPC_SUCCESS) {
		LOGE(LOG_ERROR, "[rpc] %s failed.", __FUNCTION__);
        return CUDA_ERROR_UNKNOWN;
	}
    return result;
}
CUresult cuModuleGetFunction(CUfunction* hfun, CUmodule hmod, const char* name)
{
	enum clnt_stat retval;
    ptr_result result;
    kernel_info_t *info;
    retval = rpc_cumodulegetfunction_1((uint64_t)hmod, (char*)name, &result, clnt);
    printf("[rpc] %s(%p, %s) = %d, result %p\n", __FUNCTION__, hmod, name, result.err, result.ptr_result_u.ptr);
	if (retval != RPC_SUCCESS) {
		fprintf(stderr, "[rpc] %s failed.", __FUNCTION__);
        return CUDA_ERROR_UNKNOWN;
	}
    *hfun = (CUfunction)result.ptr_result_u.ptr;
    if ((info = utils_search_info(&kernel_infos, (char*)name)) == NULL) {
        LOGE(LOG_ERROR, "cannot find kernel %s kernel_info_t", name);
        return CUDA_ERROR_UNKNOWN;
    }
    info->host_fun = *hfun;
    LOGE(LOG_DEBUG, "add host_fun %p to %s", info->host_fun, info->name);
    return result.err;
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
//DEF_FN(CUresult, cuMemcpyHtoD, CUdeviceptr, dstDevice, const void*, srcHost, size_t, ByteCount)
#undef cuMemcpyHtoD
CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount)
{
	enum clnt_stat retval;
    mem_data src;
    int result;
    src.mem_data_len = ByteCount;
    src.mem_data_val = (void*)srcHost;
    retval = rpc_cumemcpyhtod_1((uint64_t)dstDevice, src, &result, clnt);
    printf("[rpc] %s = %d\n", __FUNCTION__, result);
	if (retval != RPC_SUCCESS) {
		fprintf(stderr, "[rpc] %s failed.", __FUNCTION__);
        return CUDA_ERROR_UNKNOWN;
	}
    return result;
}
DEF_FN(CUresult, cuMemcpyHtoD_v2_ptds, CUdeviceptr, dstDevice, const void*, srcHost, size_t, ByteCount)
#undef cuMemcpyDtoH
DEF_FN(CUresult, cuMemcpyDtoH, void*, dstHost, CUdeviceptr, srcDevice, size_t, ByteCount)
DEF_FN(CUresult, cuMemcpyDtoH_v2_ptds, void*, dstHost, CUdeviceptr, srcDevice, size_t, ByteCount)
#undef cuMemcpyDtoD
DEF_FN(CUresult, cuMemcpyDtoD, CUdeviceptr, dstDevice, CUdeviceptr, srcDevice, size_t, ByteCount)
DEF_FN(CUresult, cuMemcpyDtoD_v2_ptds, CUdeviceptr, dstDevice, CUdeviceptr, srcDevice, size_t, ByteCount)
#undef cuMemcpyDtoA
DEF_FN(CUresult, cuMemcpyDtoA, CUarray, dstArray, size_t, dstOffset, CUdeviceptr, srcDevice, size_t, ByteCount)
#undef cuMemcpyAtoD
DEF_FN(CUresult, cuMemcpyAtoD, CUdeviceptr, dstDevice, CUarray, srcArray, size_t, srcOffset, size_t, ByteCount)
#undef cuMemcpyHtoA
DEF_FN(CUresult, cuMemcpyHtoA, CUarray, dstArray, size_t, dstOffset, const void*, srcHost, size_t, ByteCount)
#undef cuMemcpyAtoH
DEF_FN(CUresult, cuMemcpyAtoH, void*, dstHost, CUarray, srcArray, size_t, srcOffset, size_t, ByteCount)
#undef cuMemcpyAtoA
DEF_FN(CUresult, cuMemcpyAtoA, CUarray, dstArray, size_t, dstOffset, CUarray, srcArray, size_t, srcOffset, size_t, ByteCount)
#undef cuMemcpy2D
DEF_FN(CUresult, cuMemcpy2D, const CUDA_MEMCPY2D*, pCopy)
#undef cuMemcpy2DUnaligned
DEF_FN(CUresult, cuMemcpy2DUnaligned, const CUDA_MEMCPY2D*, pCopy)
DEF_FN(CUresult, cuMemcpy2DUnaligned_v2_ptds, const CUDA_MEMCPY2D*, pCopy)
#undef cuMemcpy3D
DEF_FN(CUresult, cuMemcpy3D, const CUDA_MEMCPY3D*, pCopy)
DEF_FN(CUresult, cuMemcpy3D_v2_ptds, const CUDA_MEMCPY3D*, pCopy)
DEF_FN(CUresult, cuMemcpyPeerAsync, CUdeviceptr, dstDevice, CUcontext, dstContext, CUdeviceptr, srcDevice, CUcontext, srcContext, size_t, ByteCount, CUstream, hStream)
DEF_FN(CUresult, cuMemcpyPeerAsync_ptsz, CUdeviceptr, dstDevice, CUcontext, dstContext, CUdeviceptr, srcDevice, CUcontext, srcContext, size_t, ByteCount, CUstream, hStream)
#undef cuMemcpyHtoAAsync
DEF_FN(CUresult, cuMemcpyHtoAAsync, CUarray, dstArray, size_t, dstOffset, const void*, srcHost, size_t, ByteCount, CUstream, hStream)
#undef cuMemcpyAtoHAsync
DEF_FN(CUresult, cuMemcpyAtoHAsync, void*, dstHost, CUarray, srcArray, size_t, srcOffset, size_t, ByteCount, CUstream, hStream)
DEF_FN(CUresult, cuMemcpy3DPeerAsync, const CUDA_MEMCPY3D_PEER*, pCopy, CUstream, hStream)
DEF_FN(CUresult, cuMemcpy3DPeerAsync_ptsz, const CUDA_MEMCPY3D_PEER*, pCopy, CUstream, hStream)
#undef cuMemcpyHtoDAsync
DEF_FN(CUresult, cuMemcpyHtoDAsync, CUdeviceptr, dstDevice, const void*, srcHost, size_t, ByteCount, CUstream, hStream)
DEF_FN(CUresult, cuMemcpyHtoDAsync_v2_ptsz, CUdeviceptr, dstDevice, const void*, srcHost, size_t, ByteCount, CUstream, hStream)
#undef cuMemcpyDtoHAsync
DEF_FN(CUresult, cuMemcpyDtoHAsync, void*, dstHost, CUdeviceptr, srcDevice, size_t, ByteCount, CUstream, hStream)
DEF_FN(CUresult, cuMemcpyDtoHAsync_v2_ptsz, void*, dstHost, CUdeviceptr, srcDevice, size_t, ByteCount, CUstream, hStream)
#undef cuMemcpyDtoDAsync
DEF_FN(CUresult, cuMemcpyDtoDAsync, CUdeviceptr, dstDevice, CUdeviceptr, srcDevice, size_t, ByteCount, CUstream, hStream)
DEF_FN(CUresult, cuMemcpyDtoDAsync_v2_ptsz, CUdeviceptr, dstDevice, CUdeviceptr, srcDevice, size_t, ByteCount, CUstream, hStream)
#undef cuMemcpy2DAsync
DEF_FN(CUresult, cuMemcpy2DAsync, const CUDA_MEMCPY2D*, pCopy, CUstream, hStream)
DEF_FN(CUresult, cuMemcpy2DAsync_v2_ptsz, const CUDA_MEMCPY2D*, pCopy, CUstream, hStream)
#undef cuMemcpy3DAsync
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
//DEF_FN(CUresult, cuLaunchKernel, CUfunction, f, unsigned int, gridDimX, unsigned int, gridDimY, unsigned int, gridDimZ, unsigned int, blockDimX, unsigned int, blockDimY, unsigned int, blockDimZ, unsigned int, sharedMemBytes, CUstream, hStream, void**, kernelParams, void**, extra)
CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra)
{
	enum clnt_stat retval;
    mem_data rpc_args = {0};
    int result;
    int found_kernel = 0;
    int i;
    kernel_info_t *info;
    LOGE(LOG_DEBUG, "cuLaunchKernel(%p)", f);

    for (i=0; i < kernel_infos.length; ++i) {
        if (list_at(&kernel_infos, i, (void**)&info) != 0) {
            LOGE(LOG_ERROR, "error getting element at %d", i);
            return CUDA_ERROR_INVALID_CONTEXT;
        }
        if (f != NULL && info != NULL && info->host_fun == f) {
            LOG(LOG_DEBUG, "calling kernel \"%s\" (param_size: %zd, param_num: %zd)", info->name, info->param_size, info->param_num);
            found_kernel = 1;
            break;
        }
    }

    if (!found_kernel) {
        LOGE(LOG_ERROR, "request to call unknown kernel.");
        return CUDA_ERROR_INVALID_CONTEXT;
    }


    if (kernelParams != NULL) {
        rpc_args.mem_data_len = sizeof(size_t)+info->param_num*sizeof(uint16_t)+info->param_size;
        rpc_args.mem_data_val = malloc(rpc_args.mem_data_len);
        memcpy(rpc_args.mem_data_val, &info->param_num, sizeof(size_t));
        memcpy(rpc_args.mem_data_val + sizeof(size_t), info->param_offsets, info->param_num*sizeof(uint16_t));
        for (size_t j=0, size=0; j < info->param_num; ++j) {
            size = info->param_sizes[j];
            memcpy(rpc_args.mem_data_val + sizeof(size_t) + info->param_num*sizeof(uint16_t) +
                   info->param_offsets[j],
                   kernelParams[j],
                   size);
        }
    } else if (extra != NULL) {
        LOGE(LOG_ERROR, "this way of passing kernel parameters is not yet supported");
        rpc_args.mem_data_val = extra[1];
        rpc_args.mem_data_len = (uint64_t)extra[3];
    }
    retval = rpc_culaunchkernel_1((uint64_t)f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, (uint64_t)hStream, rpc_args, &result, clnt);
    LOGE(LOG_DEBUG,"[rpc] %s = %d", __FUNCTION__, result);
	if (retval != RPC_SUCCESS) {
		LOGE(LOG_ERROR, "[rpc] %s failed.", __FUNCTION__);
        return CUDA_ERROR_UNKNOWN;
	}
    return result;
}
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
#undef cuStreamWaitValue32
DEF_FN(CUresult, cuStreamWaitValue32, CUstream, stream, CUdeviceptr, addr, cuuint32_t, value, unsigned int, flags)
DEF_FN(CUresult, cuStreamWaitValue32_ptsz, CUstream, stream, CUdeviceptr, addr, cuuint32_t, value, unsigned int, flags)
#undef cuStreamWriteValue32
DEF_FN(CUresult, cuStreamWriteValue32, CUstream, stream, CUdeviceptr, addr, cuuint32_t, value, unsigned int, flags)
DEF_FN(CUresult, cuStreamWriteValue32_ptsz, CUstream, stream, CUdeviceptr, addr, cuuint32_t, value, unsigned int, flags)
#undef cuStreamWaitValue64
DEF_FN(CUresult, cuStreamWaitValue64, CUstream, stream, CUdeviceptr, addr, cuuint64_t, value, unsigned int, flags)
DEF_FN(CUresult, cuStreamWaitValue64_ptsz, CUstream, stream, CUdeviceptr, addr, cuuint64_t, value, unsigned int, flags)
#undef cuStreamWriteValue64
DEF_FN(CUresult, cuStreamWriteValue64, CUstream, stream, CUdeviceptr, addr, cuuint64_t, value, unsigned int, flags)
DEF_FN(CUresult, cuStreamWriteValue64_ptsz, CUstream, stream, CUdeviceptr, addr, cuuint64_t, value, unsigned int, flags)
#undef cuStreamBatchMemOp
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
#undef cuIpcOpenMemHandle
DEF_FN(CUresult, cuIpcOpenMemHandle, CUdeviceptr*, pdptr, CUipcMemHandle, handle, unsigned int, Flags)
DEF_FN(CUresult, cuIpcCloseMemHandle, CUdeviceptr, dptr)
DEF_FN(CUresult, cuGraphicsUnregisterResource, CUgraphicsResource, resource)
DEF_FN(CUresult, cuGraphicsMapResources, unsigned int, count, CUgraphicsResource*, resources, CUstream, hStream)
DEF_FN(CUresult, cuGraphicsMapResources_ptsz, unsigned int, count, CUgraphicsResource*, resources, CUstream, hStream)
DEF_FN(CUresult, cuGraphicsUnmapResources, unsigned int, count, CUgraphicsResource*, resources, CUstream, hStream)
DEF_FN(CUresult, cuGraphicsUnmapResources_ptsz, unsigned int, count, CUgraphicsResource*, resources, CUstream, hStream)
DEF_FN(CUresult, cuGraphicsSubResourceGetMappedArray, CUarray*, pArray, CUgraphicsResource, resource, unsigned int, arrayIndex, unsigned int, mipLevel)
DEF_FN(CUresult, cuGraphicsResourceGetMappedMipmappedArray, CUmipmappedArray*, pMipmappedArray, CUgraphicsResource, resource)
#undef cuGraphicsResourceGetMappedPointer
DEF_FN(CUresult, cuGraphicsResourceGetMappedPointer, CUdeviceptr*, pDevPtr, size_t*, pSize, CUgraphicsResource, resource)
#undef cuGraphicsResourceSetMapFlags
DEF_FN(CUresult, cuGraphicsResourceSetMapFlags, CUgraphicsResource, resource, unsigned int, flags)
//DEF_FN(CUresult, cuGetExportTable, const void**, ppExportTable, const CUuuid*, pExportTableId)

// This function returns an array of 8 function pointers to hidden functions inside libcuda.so
/*CUresult cuGetExportTable(const void** ppExportTable, const CUuuid* pExportTableId)
{
	enum clnt_stat retval;
    char *uuid = NULL;
    ptr_result result;
    char idstr[64];
    void *orig_table;
    if (pExportTableId == NULL) {
        return CUDA_ERROR_UNKNOWN;
    }
    for (int i=0; i < 16;++i) {
        sprintf(idstr+i*3, "%02x ", pExportTableId->bytes[i] & 0xFF);
    }
    uuid = malloc(16);
    memcpy(uuid, pExportTableId->bytes, 16);
    //printf("precall %p->%p\n", ppExportTable, *ppExportTable);
    retval = rpc_cugetexporttable_1(uuid, &result, clnt);
    orig_table = (void*)result.ptr_result_u.ptr;
    LOGE(LOG_DEBUG, "[rpc] %s(%p, %s) = %d, result = %p", __FUNCTION__, ppExportTable,
                                               idstr, result.err, orig_table);

	if (retval != RPC_SUCCESS) {
		fprintf(stderr, "[rpc] %s failed.", __FUNCTION__);
        return CUDA_ERROR_UNKNOWN;
	}
    *ppExportTable = cd_client_hidden_get(orig_table);
    //printf("postcall %p->%p\n", ppExportTable, *ppExportTable);

    cd_client_hidden_incr();
    return result.err;
}*/
DEF_FN(CUresult, cuOccupancyMaxActiveBlocksPerMultiprocessor, int*, numBlocks, CUfunction, func, int, blockSize, size_t, dynamicSMemSize)
DEF_FN(CUresult, cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags, int*, numBlocks, CUfunction, func, int, blockSize, size_t, dynamicSMemSize, unsigned int, flags)
DEF_FN(CUresult, cuMemAdvise, CUdeviceptr, devPtr, size_t, count, CUmem_advise, advice, CUdevice, device)
DEF_FN(CUresult, cuMemPrefetchAsync, CUdeviceptr, devPtr, size_t, count, CUdevice, dstDevice, CUstream, hStream)
DEF_FN(CUresult, cuMemPrefetchAsync_ptsz, CUdeviceptr, devPtr, size_t, count, CUdevice, dstDevice, CUstream, hStream)
DEF_FN(CUresult, cuMemRangeGetAttribute, void*, data, size_t, dataSize, CUmem_range_attribute, attribute, CUdeviceptr, devPtr, size_t, count)
DEF_FN(CUresult, cuMemRangeGetAttributes, void**, data, size_t*, dataSizes, CUmem_range_attribute*, attributes, size_t, numAttributes, CUdeviceptr, devPtr, size_t, count)
CUresult cuGetErrorString(CUresult error, const char** pStr)
{
	enum clnt_stat retval;
    str_result result;
    result.str_result_u.str = malloc(128);
    retval = rpc_cugeterrorstring_1(error, &result, clnt);
    LOGE(LOG_DEBUG, "[rpc] %s(%d) = %d, result %s", __FUNCTION__, error, result.err, result.str_result_u.str);
	if (retval != RPC_SUCCESS) {
		fprintf(stderr, "[rpc] %s failed.", __FUNCTION__);
        return CUDA_ERROR_UNKNOWN;
	}
    if (pStr != NULL) {
       if ((*pStr = malloc(128)) != NULL) {
           strncpy((char*)(*pStr), result.str_result_u.str, 128);
        }
    }
    return result.err;
}
DEF_FN(CUresult, cuGetErrorName, CUresult, error, const char**, pStr)
DEF_FN(CUresult, cuGraphCreate, CUgraph*, phGraph, unsigned int, flags)
#undef cuGraphAddKernelNode
DEF_FN(CUresult, cuGraphAddKernelNode, CUgraphNode*, phGraphNode, CUgraph, hGraph, const CUgraphNode*, dependencies, size_t, numDependencies, const CUDA_KERNEL_NODE_PARAMS*, nodeParams)
#undef cuGraphKernelNodeGetParams
DEF_FN(CUresult, cuGraphKernelNodeGetParams, CUgraphNode, hNode, CUDA_KERNEL_NODE_PARAMS*, nodeParams)
#undef cuGraphKernelNodeSetParams
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
#if CUDA_VERSION >= 12000
#undef cuGraphInstantiate
DEF_FN(CUresult, cuGraphInstantiate, CUgraphExec*, phGraphExec, CUgraph, hGraph, unsigned long long, flags)
#else
DEF_FN(CUresult, cuGraphInstantiate, CUgraphExec*, phGraphExec, CUgraph, hGraph, CUgraphNode*, phErrorNode, char*, logBuffer, size_t, bufferSize)
#endif
DEF_FN(CUresult, cuGraphLaunch, CUgraphExec, hGraphExec, CUstream, hStream)
DEF_FN(CUresult, cuGraphLaunch_ptsz, CUgraphExec, hGraphExec, CUstream, hStream)
DEF_FN(CUresult, cuGraphExecDestroy, CUgraphExec, hGraphExec)
DEF_FN(CUresult, cuGraphDestroyNode, CUgraphNode, hNode)
DEF_FN(CUresult, cuGraphDestroy, CUgraph, hGraph)
DEF_FN(CUresult, cuGraphDestroy_ptsz, CUgraph, hGraph)
DEF_FN(CUresult, cuStreamBeginCapture_ptsz, CUstream, hStream)
#undef cuStreamBeginCapture
DEF_FN(CUresult, cuStreamBeginCapture, CUstream, hStream, CUstreamCaptureMode, mode)
DEF_FN(CUresult, cuStreamBeginCapture_v2_ptsz, CUstream, hStream)
DEF_FN(CUresult, cuStreamEndCapture, CUstream, hStream, CUgraph*, phGraph)
DEF_FN(CUresult, cuStreamEndCapture_ptsz, CUstream, hStream, CUgraph*, phGraph)
DEF_FN(CUresult, cuStreamIsCapturing, CUstream, hStream, CUstreamCaptureStatus*, captureStatus)
DEF_FN(CUresult, cuStreamIsCapturing_ptsz, CUstream, hStream, CUstreamCaptureStatus*, captureStatus)
DEF_FN(CUresult, cuThreadExchangeStreamCaptureMode, CUstreamCaptureMode*, mode)
#undef cuStreamGetCaptureInfo
DEF_FN(CUresult, cuStreamGetCaptureInfo, CUstream, hStream, CUstreamCaptureStatus*, captureStatus_out, cuuint64_t*, id_out, CUgraph*. graph_out, const CUgraphNode**, dependencies_out, size_t*, numDependencies_out)
DEF_FN(CUresult, cuStreamGetCaptureInfo_ptsz, CUstream, hStream, CUstreamCaptureStatus*, captureStatus, cuuint64_t*, id)
#undef cuGraphExecKernelNodeSetParams
DEF_FN(CUresult, cuGraphExecKernelNodeSetParams, CUgraphExec, hGraphExec, CUgraphNode, hNode, const CUDA_KERNEL_NODE_PARAMS*, nodeParams)

#if CUDA_VERSION >= 12000
#undef cuGetProcAddress
CUresult cuGetProcAddress(const char* symbol, void** pfn, int cudaVersion, cuuint64_t flags, CUdriverProcAddressQueryResult* symbolStatus) 
{
	enum clnt_stat retval;
    ptr_result result;
    LOGE(LOG_DEBUG, "%s(%s, %d, %llx)", __FUNCTION__, symbol, cudaVersion, flags);

    *pfn = elf2_symbol_address(symbol);
    if (*pfn == NULL) {
        LOGE(LOG_WARNING, "symbol %s not found.", symbol);
        return CUDA_ERROR_UNKNOWN;
    }
    // Pytorch uses the 11.3 API of this function which does not have the symbolStatus parameter
    // Because we do not support API versioning yet and to avoid segfaults, we ignore this parameter for now.
    //*symbolStatus = CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT;
    return cudaSuccess;
}
#endif



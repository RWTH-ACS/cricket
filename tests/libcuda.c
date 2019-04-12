#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"

#include "libwrap.h"

static const char* LIBCUDA_PATH = "/lib64/libcuda.so";
static void *so_handle = NULL;


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


DEF_FN(CUresult, cuInit, unsigned int, Flags)
DEF_FN(CUresult, cuDriverGetVersion, int*, driverVersion)
DEF_FN(CUresult, cuDeviceGetCount, int*, count)
DEF_FN(CUresult, cuDeviceGet, CUdevice*, device, int, ordinal)
DEF_FN(CUresult, cuCtxCreate, CUcontext*, pctx, unsigned int, flags, CUdevice, dev)
DEF_FN(CUresult, cuMemAlloc, CUdeviceptr*, dptr, size_t, bytesize)
DEF_FN(CUresult, cuMemcpyHtoD, CUdeviceptr, dstDevice, const void*, srcHost, size_t, ByteCount)
DEF_FN(CUresult, cuMemcpyDtoH, void*, dstHost, CUdeviceptr, srcDevice, size_t, ByteCount)
DEF_FN(CUresult, cuMemFree, CUdeviceptr, dptr)
DEF_FN(CUresult, cuLaunchKernel, CUfunction, f, unsigned int, gridDimX, unsigned int, gridDimY, unsigned int, gridDimZ, unsigned int, blockDimX, unsigned int, blockDimY, unsigned int, blockDimZ, unsigned int, sharedMemBytes, CUstream, hStream, void**, kernelParams, void** extra)
DEF_FN(CUresult, cuEventSynchronize, CUevent, hEvent)
DEF_FN(CUresult, cuStreamSynchronize, CUstream, hStream)
DEF_FN(CUresult, cuDevicePrimaryCtxRetain, CUcontext*, pctx, CUdevice, dev)

#include <cuda_runtime_api.h>
#include <cuda.h>

//For SHM
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "cpu_rpc_prot.h"
#include "cpu-common.h"
#include "cpu-utils.h"
#include "log.h"


bool_t cuda_malloc_1_svc(size_t argp, ptr_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cudaMalloc\n");
    result->err = cudaMalloc((void**)&result->ptr_result_u.ptr, argp);
    return 1;
}

bool_t cuda_device_synchronize_1_svc(int *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cudaDeviceSynchronize\n");
    *result = cudaDeviceSynchronize();
    return 1;
}

bool_t cuda_free_1_svc(uint64_t ptr, int *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cudaFree\n");
    *result = cudaFree((void*)ptr);
    return 1;
}

bool_t cuda_get_device_properties_1_svc(int device, mem_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cudaGetDeviceProperties(%d)\n", device);
    result->mem_result_u.data.mem_data_len = sizeof(struct cudaDeviceProp);
    result->mem_result_u.data.mem_data_val = malloc(sizeof(struct cudaDeviceProp));
    result->err = cudaGetDeviceProperties(
                      (struct cudaDeviceProp*)result->mem_result_u.data.mem_data_val,
                      device);
    return 1;
}

bool_t cuda_memcpy_htod_1_svc(uint64_t ptr, mem_data mem, size_t size, int *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cudaMemcpyHtoD\n");
    if (size != mem.mem_data_len) {
        LOGE(LOG_ERROR, "data size mismatch\n");
        return 0;
    }
    *result = cudaMemcpy((void*)ptr, mem.mem_data_val, size, cudaMemcpyHostToDevice);
    return 1;
}

bool_t cuda_memcpy_dtoh_1_svc(uint64_t ptr, size_t size, mem_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cudaMemcpyDtoH(%p, %zu)\n", ptr, size);
    result->mem_result_u.data.mem_data_len = size;
    result->mem_result_u.data.mem_data_val = malloc(size);
    result->err = cudaMemcpy(result->mem_result_u.data.mem_data_val, (void*)ptr, size, cudaMemcpyDeviceToHost);
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

bool_t cuda_get_device_count_1_svc(int_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cudaGetDeviceCount\n");
    result->err = cudaGetDeviceCount(&result->int_result_u.data);
    return 1;
}

bool_t cuda_device_get_attribute_1_svc(int attr, int device, int_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cudaDeviceGetAttribute\n");
    result->err = cudaDeviceGetAttribute(&result->int_result_u.data, (enum cudaDeviceAttr)attr, device);
    return 1;
}

bool_t cuda_set_device_1_svc(int device, int *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cudaSetDevice\n");
    *result = cudaSetDevice(device);
    return 1;
}

bool_t cuda_event_create_1_svc(ptr_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cudaEventCreate\n");
    result->err = cudaEventCreate((struct CUevent_st**)&result->ptr_result_u.ptr);
    return 1;
}

bool_t cuda_stream_create_with_flags_1_svc(int flags, ptr_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cudaStreamCreateWithFlags\n");
    result->err = cudaStreamCreateWithFlags((struct CUstream_st**)&result->ptr_result_u.ptr,
                                            flags);
    return 1;
}

bool_t cuda_stream_synchronize_1_svc(ptr stream, int *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cudaStreamSynchronize\n");
    *result = cudaStreamSynchronize((struct CUstream_st*)stream);
    return 1;
}

bool_t cuda_event_record_1_svc(ptr event, ptr stream, int *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cudaEventRecord\n");
    *result = cudaEventRecord((struct CUevent_st*) event, (struct CUstream_st*)stream);
    return 1;
}

bool_t cuda_event_elapsed_time_1_svc(ptr start, ptr end, float_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cudaEventElapsedTime\n");
    result->err = cudaEventElapsedTime(&result->float_result_u.data, (struct CUevent_st*) start, (struct CUevent_st*)end);
    return 1;
}

bool_t cuda_event_destroy_1_svc(ptr event, int *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cudaEventDestroy\n");
    *result = cudaEventDestroy((struct CUevent_st*) event);
    return 1;
}

bool_t cuda_event_synchronize_1_svc(ptr event, int *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cudaEventSynchronize\n");
    *result = cudaEventSynchronize((struct CUevent_st*) event);
    return 1;
}

bool_t cuda_get_error_name_1_svc(int error, str_result *result, struct svc_req *rqstp)
{
    const char* str;
    result->str_result_u.str = malloc(128);
    LOGE(LOG_DEBUG, "cudaGetErrorName\n");
    str = cudaGetErrorName((cudaError_t)error);
    strncpy(result->str_result_u.str, str, 128);
    result->err = 0;
    return 1;
}

bool_t cuda_host_alloc_1_svc(int shm_cnt, size_t size, ptr client_ptr, int *result, struct svc_req *rqstp)
{
    int ret = cudaErrorMemoryAllocation;
    int fd_shm;
    char shm_name[128];
    void *shm_addr;

    LOGE(LOG_DEBUG, "cudaHostAlloc");

    //Should only be supported for UNIX transport (using shm).
    //I don't see how TCP can profit from HostAlloc
    //TODO: Test the below
    if (socktype != UNIX) {
        LOGE(LOG_ERROR, "cudaHostAlloc is not supported for other transports than UNIX. This error means cricket_server and cricket_client are not compiled correctly (different transports)");
        goto out;
    }

    snprintf(shm_name, 128, "/crickethostalloc-%p", shm_cnt);
    if ((fd_shm = shm_open(shm_name, O_RDWR, 600)) == -1) {
        LOGE(LOG_ERROR, "ERROR: could not open shared memory");
        goto out;
    }
    if (ftruncate(fd_shm, size) == -1) {
        LOGE(LOG_ERROR, "ERROR: cannot resize shared memory");
        goto cleanup;
    }
    if ((shm_addr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_shm, 0)) == MAP_FAILED) {
        LOGE(LOG_ERROR, "ERROR: mmap returned unexpected pointer: %p", shm_addr);
        goto cleanup;
    }


    ret = cudaSuccess;
cleanup:
    shm_unlink(shm_name);
out:
    return ret;

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

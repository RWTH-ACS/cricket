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
#ifdef WITH_IB
#include <pthread.h>
#include "cpu-ib.h"
#endif //WITH_IB

typedef struct host_alloc_info {
    int cnt;
    size_t size;
    void *client_ptr;
    void *server_ptr;
} host_alloc_info_t;
static host_alloc_info_t hainfo[64];
static size_t hainfo_cnt = 1;

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

bool_t cuda_memcpy_dtod_1_svc(ptr dst, ptr src, size_t size, int *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cudaMemcpyDtoD\n");
    *result = cudaMemcpy((void*)dst, (void*)src, size, cudaMemcpyDeviceToDevice);
    return 1;
}

bool_t cuda_memcpy_htod_1_svc(uint64_t ptr, mem_data mem, size_t size, int *result, struct svc_req *rqstp)
{
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
    return 1;
}

bool_t cuda_memcpy_to_symbol_1_svc(uint64_t ptr, mem_data mem, size_t size, size_t offset, int *result, struct svc_req *rqstp)
{
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
    return 1;
}

bool_t cuda_memcpy_to_symbol_shm_1_svc(int index, ptr device_ptr, size_t size, size_t offset, int kind, int *result, struct svc_req *rqstp)
{
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
    return 1;
}
#endif //WITH_IB

bool_t cuda_memcpy_shm_1_svc(int index, ptr device_ptr, size_t size, int kind, int *result, struct svc_req *rqstp)
{
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
    return 1;
}

bool_t cuda_memcpy_dtoh_1_svc(uint64_t ptr, size_t size, mem_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cudaMemcpyDtoH(%p, %zu)\n", ptr, size);
    result->mem_result_u.data.mem_data_len = size;
    result->mem_result_u.data.mem_data_val = malloc(size);
#ifdef WITH_MEMCPY_REGISTER
    if ((result->err = cudaHostRegister(result->mem_result_u.data.mem_data_val,
                                        size, cudaHostRegisterMapped)) != cudaSuccess) {
        LOGE(LOG_ERROR, "cudaHostRegister failed.");
        return 1;
    }
#endif
    result->err = cudaMemcpy(result->mem_result_u.data.mem_data_val, (void*)ptr, size, cudaMemcpyDeviceToHost);
#ifdef WITH_MEMCPY_REGISTER
    cudaHostUnregister(result->mem_result_u.data.mem_data_val);
#endif
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
bool_t cuda_get_device_1_svc(int_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cudaGetDevice\n");
    result->err = cudaGetDevice(&result->int_result_u.data);
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

bool_t cuda_get_last_error_1_svc(int *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cudaGetLastError\n");
    *result = cudaGetLastError();
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

bool_t cuda_get_error_string_1_svc(int error, str_result *result, struct svc_req *rqstp)
{
    const char* str;
    result->str_result_u.str = malloc(256);
    LOGE(LOG_DEBUG, "cudaGetErrorString\n");
    str = cudaGetErrorString((cudaError_t)error);
    strncpy(result->str_result_u.str, str, 256);
    result->err = 0;
    return 1;
}


bool_t cuda_free_host_1_svc(int index, int *result, struct svc_req *rqstp)
{
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
    return 1;
}

bool_t cuda_host_alloc_1_svc(int client_cnt, size_t size, ptr client_ptr, unsigned int flags, int *result, struct svc_req *rqstp)
{
    int fd_shm;
    char shm_name[128];
    void *shm_addr;
    unsigned int register_flags = 0;
    *result = cudaErrorMemoryAllocation;

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
    return 1;

}

bool_t cuda_stream_destroy_1_svc(ptr stream, int *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cudaStreamDestroy\n");
    *result = cudaStreamDestroy((cudaStream_t) stream);
    return 1;
}

bool_t cuda_device_reset_1_svc(int *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cudaDeviceReset\n");
    *result = cudaDeviceReset();
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

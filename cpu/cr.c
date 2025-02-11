#define _GNU_SOURCE
#include <arpa/inet.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <errno.h>
#include <fcntl.h>
#include <rpc/clnt.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "api-recorder.h"
#include "resource-mg.h"
#include "log.h"
#include "cpu_rpc_prot.h"
#include "list.h"

static int cr_restore_array_3d(const char *path, api_record_t *record, size_t *mem_size, void *mem_data, uint64_t result_ptr, void **cuda_ptr)
{
    char *file_name;
    FILE *fp = NULL;
    const char *suffix = "array";
    cuda_malloc_3d_array_1_argument *arg;
    cuda_channel_format_desc desc;
    struct cudaExtent extent;
    struct cudaChannelFormatDesc cudaDesc;
    int flags;
    int ret = 1;

    arg = (cuda_malloc_3d_array_1_argument*)record->arguments;
    desc = arg->arg1;
    extent.depth = arg->arg2;
    extent.height = arg->arg3;
    extent.width = arg->arg4;
    flags = arg->arg5;
    cudaDesc.f = desc.f;
    cudaDesc.w = desc.w;
    cudaDesc.x = desc.x;
    cudaDesc.y = desc.y;
    cudaDesc.z = desc.z;
    *mem_size = extent.width * extent.height * extent.depth;

    if ( (mem_data = malloc(*mem_size)) == NULL) {
        LOGE(LOG_ERROR, "could not allocate memory");
        return 0;
    }

    if (asprintf(&file_name, "%s/%s-0x%lx",
                 path, suffix, result_ptr) < 0) {
        LOGE(LOG_ERROR, "memory allocation failed");
        goto out;
    }

    if ((fp = fopen(file_name, "rb")) == NULL) {
        LOGE(LOG_ERROR, "error while opening file");
        free(file_name);
        goto out;
    }

    if (ferror(fp) || feof(fp)) {
        LOGE(LOG_ERROR, "file descriptor is invalid");
        goto cleanup;
    }

    if (fread(mem_data,
               1, *mem_size, fp) != *mem_size) {
        LOGE(LOG_ERROR, "error reading mem_data");
        goto cleanup;
    }

    if ( (ret = cudaMalloc3DArray((cudaArray_t*)cuda_ptr, &cudaDesc,
                                extent, flags)) != cudaSuccess) {
        LOGE(LOG_ERROR, "cudaMalloc3DArray returned an error: %s", cudaGetErrorString(ret));
        return 1;
    }
cleanup:
    free(file_name);
    fclose(fp);
out:
    return ret;
}

static int cr_restore_array_1d(const char *path, api_record_t *record, size_t *mem_size, void *mem_data, uint64_t result_ptr, void **cuda_ptr)
{
    char *file_name;
    FILE *fp = NULL;
    const char *suffix = "array";
    cuda_malloc_array_1_argument *arg;
    cuda_channel_format_desc desc;
    struct cudaChannelFormatDesc cudaDesc;
    size_t width;
    size_t height;
    int flags;
    int ret = 1;

    arg = (cuda_malloc_array_1_argument*)record->arguments;
    desc = arg->arg1;
    width = arg->arg2;
    height = arg->arg3;
    flags = arg->arg4;
    cudaDesc.f = desc.f;
    cudaDesc.w = desc.w;
    cudaDesc.x = desc.x;
    cudaDesc.y = desc.y;
    cudaDesc.z = desc.z;
    *mem_size = width * height;

    if ( (mem_data = malloc(*mem_size)) == NULL) {
        LOGE(LOG_ERROR, "could not allocate memory");
        return 0;
    }

    if (asprintf(&file_name, "%s/%s-0x%lx",
                 path, suffix, result_ptr) < 0) {
        LOGE(LOG_ERROR, "memory allocation failed");
        goto out;
    }

    if ((fp = fopen(file_name, "rb")) == NULL) {
        LOGE(LOG_ERROR, "error while opening file");
        free(file_name);
        goto out;
    }

    if (ferror(fp) || feof(fp)) {
        LOGE(LOG_ERROR, "file descriptor is invalid");
        goto cleanup;
    }

    if (fread(mem_data,
               1, *mem_size, fp) != *mem_size) {
        LOGE(LOG_ERROR, "error reading mem_data");
        goto cleanup;
    }

    if ( (ret = cudaMallocArray((cudaArray_t*)cuda_ptr, &cudaDesc,
                                width, height, flags)) != cudaSuccess) {
        LOGE(LOG_ERROR, "cudaMallocArray returned an error: %s", cudaGetErrorString(ret));
        return 1;
    }
cleanup:
    free(file_name);
    fclose(fp);
out:
    return ret;
}

static int cr_restore_arrays(const char *path, api_record_t *record, resource_mg *rm_arrays)
{
    size_t mem_size;
    void *mem_data;
    void *cuda_ptr;
    ptr_result result;
    int ret;
    if (record->function != CUDA_MALLOC_ARRAY ||
        record->function != CUDA_MALLOC_3D_ARRAY) {
        LOGE(LOG_ERROR, "got a record that is not of type cudaMallocArray");
        return 0;
    }
    result = record->result.ptr_result_u;

    if (record->function == CUDA_MALLOC_ARRAY) {
        if (cr_restore_array_1d(path, record, &mem_size,
                                mem_data, result.ptr_result_u.ptr, &cuda_ptr) != 0) {
            LOGE(LOG_ERROR, "error restoring 1D array");
            goto out;
        }
    } else if (record->function == CUDA_MALLOC_3D_ARRAY) {
        if (cr_restore_array_3d(path, record, &mem_size,
                                mem_data, result.ptr_result_u.ptr, &cuda_ptr) != 0) {
            LOGE(LOG_ERROR, "error restoring 1D array");
            goto out;
        }
    }

    LOG(LOG_DEBUG, "restored array mapping %p -> %p",
                               (void*)result.ptr_result_u.ptr,
                               cuda_ptr);

    if (resource_mg_add_sorted(rm_arrays,
                              (void*)result.ptr_result_u.ptr,
                              cuda_ptr) != 0) {
        LOGE(LOG_ERROR, "error adding memory resource to resource manager");
        return 1;
    }

    if ( (ret = cudaMemcpy(cuda_ptr,
           mem_data,
           mem_size,
           cudaMemcpyHostToDevice)) != 0) {
        LOGE(LOG_ERROR, "cudaMalloc returned an error: %s", cudaGetErrorString(ret));
        return 0;
    }
    LOG(LOG_DEBUG, "restored memory of size %zu from ptr 0x%lx", mem_size, result.ptr_result_u.ptr);
    ret = 0;
out:
    return ret;
}

static int cr_restore_cusolver(api_record_t *record, resource_mg *rm_cusolver)
{
    cusolverDnHandle_t new_handle = NULL;
    cusolverStatus_t err;
    ptr_result res = record->result.ptr_result_u;
    if (record->function == rpc_cusolverDnCreate) {
        if ((err = cusolverDnCreate(&new_handle)) != CUSOLVER_STATUS_SUCCESS) {
            LOGE(LOG_ERROR, "CUSOLVER error while restoring event");
            return 1;
        }
    } else {
        LOGE(LOG_ERROR, "cannot restore an cusolver handle from a record that is not an cusolver_create record");
        return 1;
    }
    if (resource_mg_add_sorted(rm_cusolver, (void*)res.ptr_result_u.ptr, new_handle) != 0) {
        LOGE(LOG_ERROR, "error adding to event resource manager");
        return 1;
    }
    return 0;
}

static int cr_restore_cublas(api_record_t *record, resource_mg *rm_cublas)
{
    cublasHandle_t new_handle = NULL;
    cublasStatus_t err;
    ptr_result res = record->result.ptr_result_u;
    if (record->function == rpc_cublasCreate) {
        if ((err = cublasCreate_v2(&new_handle)) != CUBLAS_STATUS_SUCCESS) {
            LOGE(LOG_ERROR, "cublas error while restoring event");
            return 1;
        }
    } else {
        LOGE(LOG_ERROR, "cannot restore a cublas handle from a record that is not a cublas_create record");
        return 1;
    }
    if (resource_mg_add_sorted(rm_cublas, (void*)res.ptr_result_u.ptr, new_handle) != 0) {
        LOGE(LOG_ERROR, "error adding to event resource manager");
        return 1;
    }
    return 0;
}

static int cr_restore_device(api_record_t *record, resource_mg *rm_devices)
{
    ptr device;
    ptr_result res = record->result.ptr_result_u;
    if (record->function != rpc_cuDeviceGet && record->arg_size != sizeof(int)) {
        LOGE(LOG_ERROR, "got a record that is not of type rpc_cudeviceget or unexpected argument size");
        return 0;
    }
    int ordinal = *(int*)record->arguments;
    CUresult err = cuDeviceGet((CUdevice*)&device, ordinal);
    if (err != CUDA_SUCCESS) {
        LOGE(LOG_ERROR, "cuDeviceGet failed: %d", res);
        return 1;
    }
    if (resource_mg_add_sorted(rm_devices, (void*)res.ptr_result_u.ptr, (void*)device) != 0) {
        LOGE(LOG_ERROR, "error adding to event resource manager");
        return 1;
    }
    return 0;
}

static int cr_restore_events(api_record_t *record, resource_mg *rm_events)
{
    cudaEvent_t new_event = NULL;
    cudaError_t err;
    ptr_result res = record->result.ptr_result_u;
    if (record->function == CUDA_EVENT_CREATE) {
        if ((err = cudaEventCreate(&new_event)) != cudaSuccess) {
            LOGE(LOG_ERROR, "CUDA error while restoring event: %s", cudaGetErrorString(err));
            return 1;
        }
    } else if (record->function == CUDA_EVENT_CREATE_WITH_FLAGS) {
        if ((err = cudaEventCreateWithFlags(&new_event, *(int*)record->arguments)) != cudaSuccess) {
            LOGE(LOG_ERROR, "CUDA error while restoring event: %s", cudaGetErrorString(err));
            return 1;
        }
    } else {
        LOGE(LOG_ERROR, "cannot restore an event from a record that is not an event_create record");
        return 1;
    }
    if (resource_mg_add_sorted(rm_events, (void*)res.ptr_result_u.ptr, new_event) != 0) {
        LOGE(LOG_ERROR, "error adding to event resource manager");
        return 1;
    }
    return 0;
}

static int cr_restore_streams(api_record_t *record, resource_mg *rm_streams)
{
    cudaStream_t new_stream = NULL;
    cudaError_t err;
    ptr_result res = record->result.ptr_result_u;
    if (record->function == CUDA_STREAM_CREATE) {
        if ((err = cudaStreamCreate(&new_stream)) != cudaSuccess) {
            LOGE(LOG_ERROR, "CUDA error while restoring stream: %s", cudaGetErrorString(err));
            return 1;
        }
    } else if (record->function == CUDA_STREAM_CREATE_WITH_FLAGS) {
        if ((err = cudaStreamCreateWithFlags(&new_stream, *(int*)record->arguments)) != cudaSuccess) {
            LOGE(LOG_ERROR, "CUDA error while restoring stream: %s", cudaGetErrorString(err));
            return 1;
        }
    } else if (record->function == CUDA_STREAM_CREATE_WITH_PRIORITY) {
        cuda_stream_create_with_priority_1_argument *arg = record->arguments;
        if ((err = cudaStreamCreateWithPriority(&new_stream, arg->arg1, arg->arg2)) != cudaSuccess) {
            LOGE(LOG_ERROR, "CUDA error while restoring stream: %s", cudaGetErrorString(err));
            return 1;
        }
    } else {
        LOGE(LOG_ERROR, "cannot restore a stream from a record that is not a stream_create record");
        return 1;
    }
    if (resource_mg_add_sorted(rm_streams, (void*)res.ptr_result_u.ptr, new_stream) != 0) {
        LOGE(LOG_ERROR, "error adding to stream resource manager");
        return 1;
    }
    LOGE(LOG_DEBUG, "add stream mapping %p->%p", (void*)res.ptr_result_u.ptr, new_stream);
    return 0;
}

static int cr_restore_elf(const char *path, api_record_t *record,
                          resource_mg *rm_modules)
{
    FILE *fp = NULL;
    char *file_name;
    int elf_restored = 1;
    const char *suffix = "elf";
    void *mem_data;
    rpc_elf_load_1_argument *arg;
    int result;
    CUresult res;
    CUmodule module = NULL;
    if (record->function != rpc_elf_load) {
        LOGE(LOG_ERROR, "got a record that is not of type rpc_elf_load");
        return 0;
    }
    arg = (rpc_elf_load_1_argument *)record->arguments;
    result = record->result.integer;
    if ((mem_data = malloc(arg->arg1.mem_data_len)) == NULL) {
        LOGE(LOG_ERROR, "could not allocate memory");
        return 1;
    }

    if (asprintf(&file_name, "%s/%s-0x%lx", path, suffix, arg->arg2) < 0) {
        LOGE(LOG_ERROR, "memory allocation failed");
        goto out;
    }

    if ((fp = fopen(file_name, "rb")) != NULL) {

        if (ferror(fp) || feof(fp)) {
            LOGE(LOG_ERROR, "file descriptor is invalid");
            goto cleanup;
        }

        if (fread(mem_data, 1, arg->arg1.mem_data_len, fp) !=
            arg->arg1.mem_data_len) {
            LOGE(LOG_ERROR, "error reading mem_data");
            goto cleanup;
        }
    } else {
        LOGE(LOG_WARNING, "could not open memory file: %s", file_name);
        elf_restored = 0;
    }

    if ((res = cuModuleLoadData(&module, mem_data)) != CUDA_SUCCESS) {
        LOGE(LOG_ERROR, "cuModuleLoadData failed: %d", res);
        result = res;
        goto cleanup;
    }

    LOG(LOG_DEBUG, "restored elf %p -> %p", arg->arg2, module);

    if ((res = resource_mg_add_sorted(rm_modules, (void *)arg->arg2,
                                      (void *)module)) != 0) {
        LOGE(LOG_ERROR, "resource_mg_create failed: %d", res);
        result = res;
        goto cleanup;
    }

cleanup:
    free(file_name);
    fclose(fp);
out:
    return result;
}

static int cr_restore_memory(const char *path, api_record_t *record, memory_mg *rm_memory)
{
    FILE *fp = NULL;
    char *file_name;
    const char *suffix = "mem";
    size_t mem_size;
    void *mem_data;
    void *cuda_ptr;
    ptr_result result;
    int ret;
    int restore_memory = 1;
    if (record->function != CUDA_MALLOC) {
        LOGE(LOG_ERROR, "got a record that is not of type cudaMalloc");
        return 1;
    }
    mem_size = *(size_t*)record->arguments;
    result = record->result.ptr_result_u;

    if ( (mem_data = malloc(mem_size)) == NULL) {
        LOGE(LOG_ERROR, "could not allocate memory");
        return 1;
    }

    if (asprintf(&file_name, "%s/%s-0x%lx",
                 path, suffix, result.ptr_result_u.ptr) < 0) {
        LOGE(LOG_ERROR, "memory allocation failed");
        goto out;
    }

    if ((fp = fopen(file_name, "rb")) != NULL) {

        if (ferror(fp) || feof(fp)) {
            LOGE(LOG_ERROR, "file descriptor is invalid");
            goto cleanup;
            return 1;
        }

        if (fread(mem_data,
                   1, mem_size, fp) != mem_size) {
            LOGE(LOG_ERROR, "error reading mem_data");
            goto cleanup;
        }
    } else {
        LOGE(LOG_WARNING, "could not open memory file: %s", file_name);
        restore_memory = 0;
    }

    if ( (ret = cudaMalloc(&cuda_ptr, mem_size)) != cudaSuccess) {
        LOGE(LOG_ERROR, "cudaMalloc returned an error: %s", cudaGetErrorString(ret));
        return 1;
    }

    LOG(LOG_DEBUG, "restored mapping %p -> %p", 
                               (void*)result.ptr_result_u.ptr, 
                               cuda_ptr);

    if (memory_mg_add_sorted(rm_memory, 
                               (void*)result.ptr_result_u.ptr, 
                               cuda_ptr, mem_size) != 0) {
        LOGE(LOG_ERROR, "error adding memory resource to resource manager");
        return 1;
    }

    if ( restore_memory == 1 && 
         (ret = cudaMemcpy(cuda_ptr,
           mem_data,
           mem_size,
           cudaMemcpyHostToDevice)) != 0) {
        LOGE(LOG_ERROR, "cudaMalloc returned an error: %s", cudaGetErrorString(ret));
        return 1;
    }
    LOG(LOG_DEBUG, "restored memory of size %zu from %s", mem_size, file_name);
    ret = 0;
cleanup:
    free(file_name);
    if (restore_memory == 1) {
        fclose(fp);
    }
out:
    return ret;
}

static int cr_dump_elf_entry(const char *path, api_record_t *record)
{
    FILE *fp = NULL;
    char *file_name;
    const char *suffix = "elf";
    rpc_elf_load_1_argument *arg;
    int result;
    int ret;
    if (record->function != rpc_elf_load) {
        LOGE(LOG_ERROR, "got a record that is not of type rpc_elf_load");
        return 0;
    }
    arg = (rpc_elf_load_1_argument *)record->arguments;
    result = record->result.integer;

    if (asprintf(&file_name, "%s/%s-0x%lx", path, suffix, arg->arg2) < 0) {
        LOGE(LOG_ERROR, "memory allocation failed");
        goto out;
    }
    if ((fp = fopen(file_name, "w+b")) == NULL) {
        LOGE(LOG_ERROR, "error while opening file");
        free(file_name);
        goto out;
    }

    if (fwrite(arg->arg1.mem_data_val, 1, arg->arg1.mem_data_len, fp) !=
        arg->arg1.mem_data_len) {
        LOGE(LOG_ERROR, "error writing mem_data");
        return 1;
    }
    LOG(LOG_DEBUG, "dumped elf from %#x of size %zu to %s",
        arg->arg1.mem_data_val, arg->arg1.mem_data_len, file_name);
    ret = 0;
cleanup:
    free(file_name);
    fclose(fp);
out:
    return ret;
}

static int cr_dump_memory_entry(const char *path, api_record_t *record)
{
    FILE *fp = NULL;
    char *file_name;
    const char *suffix = "mem";
    size_t mem_size;
    void *mem_data;
    ptr_result result;
    int ret;
    if (record->function != CUDA_MALLOC) {
        LOGE(LOG_ERROR, "got a record that is not of type cudaMalloc");
        return 0;
    }
    mem_size = *(size_t*)record->arguments;
    result = record->result.ptr_result_u;

    if ( (mem_data = malloc(mem_size)) == NULL) {
        LOGE(LOG_ERROR, "could not allocate memory");
        return 0;
    }

    if ( (ret = cudaMemcpy(mem_data,
           (void*)result.ptr_result_u.ptr,
           mem_size,
           cudaMemcpyDeviceToHost)) != 0) {
        LOGE(LOG_ERROR, "cudaMemcpy returned an error: %s",
             cudaGetErrorString(ret));
        return 0;
    }

    if (asprintf(&file_name, "%s/%s-0x%lx",
                 path, suffix, result.ptr_result_u.ptr) < 0) {
        LOGE(LOG_ERROR, "memory allocation failed");
        goto out;
    }
    if ((fp = fopen(file_name, "w+b")) == NULL) {
        LOGE(LOG_ERROR, "error while opening file");
        free(file_name);
        goto out;
    }

    if (fwrite(mem_data,
               1, mem_size, fp) != mem_size) {
        LOGE(LOG_ERROR, "error writing mem_data");
        return 1;
    }
    LOG(LOG_DEBUG, "dumped memory of size %zu to %s", mem_size, file_name);
    ret = 0;
cleanup:
    free(file_name);
    fclose(fp);
out:
    return ret;
}

static int cr_dump_api_record(FILE *fp, api_record_t *record)
{
    size_t size;
    int ret = 1;
    size = sizeof(record->function);
    if (fwrite(&record->function,
               1, size, fp) != size) {
        LOGE(LOG_ERROR, "error writing record->function");
        goto cleanup;
    }
    size = sizeof(record->arg_size);
    if (fwrite(&record->arg_size,
               1, size, fp) != size) {
        LOGE(LOG_ERROR, "error writing record->arg_size");
        goto cleanup;
    }
    size = record->arg_size;
    if (fwrite(record->arguments,
               1, size, fp) != size) {
        LOGE(LOG_ERROR, "error writing record->arguments");
        goto cleanup;
    }
    size = sizeof(record->result);
    if (fwrite(&record->result,
               1, size, fp) != size) {
        LOGE(LOG_ERROR, "error writing record->result");
        goto cleanup;
    }
    size = sizeof(record->data_size);
    if (fwrite(&record->data_size,
               1, size, fp) != size) {
        LOGE(LOG_ERROR, "error writing record->data_size");
        goto cleanup;
    }
    size = record->data_size;
    if (size > 0 && record->data != NULL) {
        if (fwrite(record->data,
                   1, size, fp) != size) {
            LOGE(LOG_ERROR, "error writing record->data");
            goto cleanup;
        }
    }
    ret = 0;
 cleanup:
    return ret;
}

int cr_dump_elfs(const char *path)
{
    int res = 1;
    api_record_t *record;
    LOG(LOG_DEBUG, "dumping elf records to %s", path);
    for (size_t i = 0; i < api_records.length; i++) {
        if (list_at(&api_records, i, (void **)&record) != 0) {
            LOGE(LOG_ERROR, "list_at %zu returned an error.", i);
            goto cleanup;
        }
        if (record->function == rpc_elf_load) {
            api_records_print_records(record);
            if (cr_dump_elf_entry(path, record) != 0) {
                LOGE(LOG_ERROR, "error dumping memory");
                goto cleanup;
            }
        }
    }
    res = 0;
cleanup:
    return res;
}

int cr_dump_memory(const char *path)
{
    int res = 1;
    api_record_t *record;
    LOG(LOG_DEBUG, "dumping memory records to %s", path);
    for (size_t i = 0; i < api_records.length; i++) {
        if (list_at(&api_records, i, (void**)&record) != 0) {
            LOGE(LOG_ERROR, "list_at %zu returned an error.", i);
            goto cleanup;
        }
        if (record->function == CUDA_MALLOC) {
            api_records_print_records(record);
            if (cr_dump_memory_entry(path, record) != 0) {
                LOGE(LOG_ERROR, "error dumping memory");
                goto cleanup;
            }
        }
    }
    res = 0;
cleanup:
    return res;
}

int cr_restore_rpc_id(const char *path, unsigned long *prog, unsigned long *vers)
{
    FILE *fp = NULL;
    char *file_name;
    const char *suffix = "rpc_id";
    int res = 1;
    size_t size;
    if (prog == NULL || vers == NULL) {
        goto out;
    }
    if (asprintf(&file_name, "%s/%s", path, suffix) < 0) {
        LOGE(LOG_ERROR, "memory allocation failed");
        goto out;
    }
    if ((fp = fopen(file_name, "rb")) == NULL) {
        LOGE(LOG_ERROR, "error while opening file \"%s\": %s", file_name, strerror(errno));
        free(file_name);
        goto out;
    }
    if (ferror(fp)) {
        LOGE(LOG_ERROR, "file descriptor is invalid");
        goto cleanup;
    }
    LOG(LOG_DEBUG, "restoring rpc_id from %s", file_name);
    size = sizeof(prog);
    if (fread(prog,
               1, size, fp) != size) {
        LOGE(LOG_ERROR, "error reading prog");
        goto cleanup;
    }
    size = sizeof(vers);
    if (fread(vers,
               1, size, fp) != size) {
        LOGE(LOG_ERROR, "error reading vers");
        goto cleanup;
    }

    res = 0;
cleanup:
    free(file_name);
    fclose(fp);
out:
    return res;
}

int cr_dump_rpc_id(const char *path, unsigned long prog, unsigned long vers)
{
    FILE *fp = NULL;
    char *file_name;
    const char *suffix = "rpc_id";
    int res = 1;
    size_t size;
    if (asprintf(&file_name, "%s/%s", path, suffix) < 0) {
        LOGE(LOG_ERROR, "memory allocation failed");
        goto out;
    }
    if ((fp = fopen(file_name, "w+b")) == NULL) {
        LOGE(LOG_ERROR, "error while opening file");
        free(file_name);
        goto out;
    }
    if (ferror(fp)) {
        LOGE(LOG_ERROR, "file descriptor is invalid");
        goto cleanup;
    }
    LOG(LOG_DEBUG, "dumping rpc_id to %s", file_name);
    size = sizeof(prog);
    if (fwrite(&prog,
               1, size, fp) != size) {
        LOGE(LOG_ERROR, "error writing prog");
        goto cleanup;
    }
    size = sizeof(vers);
    if (fwrite(&vers,
               1, size, fp) != size) {
        LOGE(LOG_ERROR, "error writing vers");
        goto cleanup;
    }

    res = 0;
cleanup:
    free(file_name);
    fclose(fp);
out:
    return res;
}

int cr_dump(const char *path)
{
    FILE *fp = NULL;
    char *file_name;
    const char *suffix = "api_records";
    int res = 1;
    api_record_t *record;
    if (asprintf(&file_name, "%s/%s", path, suffix) < 0) {
        LOGE(LOG_ERROR, "memory allocation failed");
        goto out;
    }
    if ((fp = fopen(file_name, "w+b")) == NULL) {
        LOGE(LOG_ERROR, "error while opening file");
        free(file_name);
        goto out;
    }
    if (ferror(fp)) {
        LOGE(LOG_ERROR, "file descriptor is invalid");
        goto cleanup;
    }
    LOG(LOG_DEBUG, "dumping api records to %s", file_name);
    for (size_t i = 0; i < api_records.length; i++) {
        if (list_at(&api_records, i, (void**)&record) != 0) {
            LOGE(LOG_ERROR, "list_at %zu returned an error.", i);
            goto cleanup;
        }
        if (cr_dump_api_record(fp, record) != 0) {
            LOGE(LOG_ERROR, "error dumping api record");
            goto cleanup;
        }
        api_records_print_records(record);
    }
    res = 0;
cleanup:
    free(file_name);
    fclose(fp);
out:
    return res;
}

static int cr_restore_api_record(FILE *fp, api_record_t *record)
{
    size_t size;
    int res = 1;
    size = sizeof(record->function);
    if (fread(&record->function,
               1, size, fp) != size) {
        if (feof(fp)) {
            res = 2;
            goto cleanup;
        }
        LOGE(LOG_ERROR, "error reading record->function");
        goto cleanup;
    }
    size = sizeof(record->arg_size);
    if (fread(&record->arg_size,
               1, size, fp) != size) {
        LOGE(LOG_ERROR, "error reading record->function");
        goto cleanup;
    }
    size = record->arg_size;
    if ( (record->arguments = malloc(size)) == NULL) {
        LOGE(LOG_ERROR, "malloc failed");
        goto cleanup;
    }
    if (fread(record->arguments,
               1, size, fp) != size) {
        LOGE(LOG_ERROR, "error reading record->function");
        goto cleanup;
    }
    size = sizeof(record->result);
    if (fread(&record->result,
               1, size, fp) != size) {
        LOGE(LOG_ERROR, "error reading record->function");
        goto cleanup;
    }
    size = sizeof(record->data_size);
    if (fread(&record->data_size,
               1, size, fp) != size) {
        LOGE(LOG_ERROR, "error reading record->data_size");
        goto cleanup;
    }
    size = record->data_size;
    if (size > 0) {
        if ( (record->data = malloc(size)) == NULL) {
            LOGE(LOG_ERROR, "malloc failed");
            goto cleanup;
        }
        if (fread(record->data,
                   1, size, fp) != size) {
            LOGE(LOG_ERROR, "error reading record->data");
            goto cleanup;
        }
    } else {
        record->data = NULL;
    }
    
    res = 0;
cleanup:
    return res;
}

extern void rpc_dispatch(struct svc_req *rqstp, xdrproc_t *ret_arg, xdrproc_t *ret_res, size_t *res_sz, bool_t (**ret_fun)(char *, void *, struct svc_req *));

int cr_call_record(api_record_t *record)
{
    struct svc_req rqstp;
    SVCXPRT rq_xprt = {.xp_fd = 0};
    xdrproc_t arg;
    xdrproc_t res;
    size_t res_sz;
    bool_t retval;
    void *result;
    bool_t (*fun)(char *, void *, struct svc_req *);
    rqstp.rq_xprt = &rq_xprt;

    rqstp.rq_proc = record->function;
    rpc_dispatch(&rqstp, &arg, &res, &res_sz, &fun);
    result = malloc(res_sz);
	retval = (bool_t) (*fun)((char *)record->arguments, result, &rqstp);

    return (retval==1 ? 0 : 1);
}

static int cr_restore_resources(const char *path, api_record_t *record,
                                memory_mg *rm_memory, resource_mg *rm_streams,
                                resource_mg *rm_events, resource_mg *rm_arrays,
                                resource_mg *rm_cusolver,
                                resource_mg *rm_cublas, resource_mg *rm_modules,
                                resource_mg *rm_devices)
{
    int ret = 1;
    switch (record->function) {
    case CUDA_MALLOC:
        if (cr_restore_memory(path, record, rm_memory) != 0) {
            LOGE(LOG_ERROR, "error dumping memory");
            goto cleanup;
        }
        break;
    case CUDA_MEMCPY_HTOD:
    case CUDA_MEMCPY_DTOD:
    case CUDA_MEMCPY_DTOH:
    case CUDA_MEMCPY_TO_SYMBOL:
    case CUDA_MEMSET:
        break;
    case rpc_elf_load:
        if (cr_restore_elf(path, record, rm_modules) != 0) {
            LOGE(LOG_ERROR, "error restoring elf");
            goto cleanup;
        }
        break;
    case rpc_register_function:
        ((rpc_register_function_1_argument *)record->arguments)->arg3 =
            record->data;
        ((rpc_register_function_1_argument *)record->arguments)->arg4 =
            record->data + strlen(record->data) + 1;

        if (cr_call_record(record) != 0) {
            LOGE(LOG_ERROR, "calling record function failed");
            goto cleanup;
        }
        break;
    case rpc_register_var:
        ((rpc_register_var_1_argument *)record->arguments)->arg4 = record->data;

        if (cr_call_record(record) != 0) {
            LOGE(LOG_ERROR, "calling record function failed");
            goto cleanup;
        }
        break;
    case CUDA_STREAM_CREATE:
    case CUDA_STREAM_CREATE_WITH_FLAGS:
    case CUDA_STREAM_CREATE_WITH_PRIORITY:
        if (cr_restore_streams(record, rm_streams) != 0) {
            LOGE(LOG_ERROR, "error restoring streams");
            goto cleanup;
        }
        break;
    case CUDA_EVENT_CREATE:
    case CUDA_EVENT_CREATE_WITH_FLAGS:
        if (cr_restore_events(record, rm_events) != 0) {
            LOGE(LOG_ERROR, "error restoring events");
            goto cleanup;
        }
        break;
    case CUDA_MALLOC_ARRAY:
    case CUDA_MALLOC_3D_ARRAY:
        if (cr_restore_arrays(path, record, rm_arrays) != 0) {
            LOGE(LOG_ERROR, "error restoring arrays");
            goto cleanup;
        }
        break;
    case CUDA_LAUNCH_KERNEL:
    case CUDA_LAUNCH_COOPERATIVE_KERNEL:
        break;
    case rpc_cusolverDnCreate:
        if (cr_restore_cusolver(record, rm_cusolver) != 0) {
            LOGE(LOG_ERROR, "error restoring cusolver");
            goto cleanup;
        }
        break;
    case rpc_cublasCreate:
        if (cr_restore_cublas(record, rm_cublas) != 0) {
            LOGE(LOG_ERROR, "error restoring cublas");
            goto cleanup;
        }
        break;
    case rpc_cuDeviceGet:
        if (cr_restore_device(record, rm_devices) != 0) {
            LOGE(LOG_ERROR, "error restoring device");
            goto cleanup;
        }
        break;
    case rpc_cuCtxCreate_v3:
    case rpc_cuctxdestroy:
        // ignore for now
        break;
    default:
        if (cr_call_record(record) != 0) {
            LOGE(LOG_ERROR, "calling record function failed");
            goto cleanup;
        }
    }
    ret = 0;
 cleanup:
    return ret;
}

int cr_launch_kernel(void)
{
    api_record_t *record;
    size_t arg_num;
    uint64_t **arg_ptr = NULL;
    static uint64_t zero = 0LL;
    int ret = 1;

    for (size_t i = api_records.length-1; i > 0; --i) {
        record = list_get(&api_records, i);
        if (record->function == CUDA_LAUNCH_KERNEL) {
            cuda_launch_kernel_1_argument *arg = 
              ((cuda_launch_kernel_1_argument*)record->arguments);
            arg->arg4.mem_data_len = record->data_size;
            arg->arg4.mem_data_val = record->data;
            if ((ret = cr_call_record(record)) != 0) {
                LOGE(LOG_ERROR, "calling record function failed");
                goto cleanup;
            }
            //free(record->data);
            /*dim3 cuda_gridDim = {arg.arg2.x, arg.arg2.y, arg.arg2.z};
            dim3 cuda_blockDim = {arg.arg3.x, arg.arg3.y, arg.arg3.z};
            LOGE(LOG_DEBUG, "launching kernel with address %p", arg.arg1);

            ret = cudaLaunchKernel(
              (void*)arg.arg1,
              cuda_gridDim,
              cuda_blockDim,
              (void**)arg_ptr,
              arg.arg5,
              (void*)arg.arg6);
            //TODO: use resource manager for stream!
            if (ret != cudaSuccess) {
                LOGE(LOG_ERROR, "error launching kernel");
            } else {
                ret = 0;
            }*/
            ret = 0;
            goto cleanup;
        } else if (record->function == CUDA_LAUNCH_COOPERATIVE_KERNEL) {
            LOGE(LOG_ERROR, "not yet supported");
            goto cleanup;
        }
    }
    ret = 0;
 cleanup:
    free(arg_ptr);
    return ret;
}

int cr_restore(const char *path, memory_mg *rm_memory,
               resource_mg *rm_streams, resource_mg *rm_events,
               resource_mg *rm_arrays, resource_mg *rm_cusolver,
               resource_mg *rm_cublas, resource_mg *rm_modules,
               resource_mg *rm_devices)
{
    FILE *fp = NULL;
    char *file_name;
    const char *suffix = "api_records";
    int res = 1;
    api_record_t *record;
    int function;
    if (asprintf(&file_name, "%s/%s", path, suffix) < 0) {
        LOGE(LOG_ERROR, "memory allocation failed");
        goto out;
    }
    if ((fp = fopen(file_name, "rb")) == NULL) {
        LOGE(LOG_ERROR, "error while opening file \"%s\": %s", file_name, strerror(errno));
        free(file_name);
        goto out;
    }
    if (ferror(fp) || feof(fp)) {
        LOGE(LOG_ERROR, "file descriptor is invalid");
        goto cleanup;
        return 1;
    }
    LOG(LOG_DEBUG, "restoring api records from %s", file_name);
    while (!feof(fp)) {
        if (list_append(&api_records, (void**)&record) != 0) {
            LOGE(LOG_ERROR, "list_append returned an error.");
            goto cleanup;
        }
        if ((res = cr_restore_api_record(fp, record)) != 0) {
            if (res == 2) {
                list_rm(&api_records, api_records.length-1);
                break;
            } else {
                LOGE(LOG_ERROR, "list_append returned an error.");
                goto cleanup;
            }
        }
        api_records_print_records(record);
        if (cr_restore_resources(path, record, rm_memory, rm_streams, rm_events,
                                 rm_arrays, rm_cusolver, rm_cublas,
                                 rm_modules, rm_devices) != 0) {
            LOGE(LOG_ERROR, "error restoring resources");
            goto cleanup;
        }
    }
    // if (cr_launch_kernel() != 0) {
    //     LOGE(LOG_ERROR, "launching kernel failed");
    //     goto cleanup;
    // }
    res = 0;
cleanup:
    free(file_name);
    fclose(fp);
out:
    return res;
}

#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <cuda_runtime.h>
#include <arpa/inet.h>
#include <rpc/clnt.h>

#include "cpu-common.h"
#include "api-recorder.h"
#include "resource-mg.h"
#include "log.h"
#include "cpu_rpc_prot.h"


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
    return 0;
}

static int cr_restore_memory(const char *path, api_record_t *record, resource_mg *rm_memory)
{
    FILE *fp = NULL;
    char *file_name;
    const char *suffix = "mem";
    size_t mem_size;
    void *mem_data;
    void *cuda_ptr;
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

    if (asprintf(&file_name, "%s/%s-0x%lx",
                 path, suffix, result.ptr_result_u.ptr) < 0) {
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
        return 1;
    }

    if (fread(mem_data,
               1, mem_size, fp) != mem_size) {
        LOGE(LOG_ERROR, "error reading mem_data");
        goto cleanup;
    }

    if ( (ret = cudaMalloc(&cuda_ptr, mem_size)) != cudaSuccess) {
        LOGE(LOG_ERROR, "cudaMalloc returned an error: %s", cudaGetErrorString(ret));
        return 1;
    }

    LOG(LOG_DEBUG, "restored mapping %p -> %p", 
                               (void*)result.ptr_result_u.ptr, 
                               cuda_ptr);

    if (resource_mg_add_sorted(rm_memory, 
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
    LOG(LOG_DEBUG, "restored memory of size %zu from %s", mem_size, file_name);
    ret = 0;
cleanup:
    free(file_name);
    fclose(fp);
out:
    return ret;
}

static int cr_dump_memory(const char *path, api_record_t *record)
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
        LOGE(LOG_ERROR, "cudaMalloc returned an error: %s", cudaGetErrorString(ret));
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
        LOGE(LOG_ERROR, "error writing record->function");
        goto cleanup;
    }
    size = record->arg_size;
    if (fwrite(record->arguments,
               1, size, fp) != size) {
        LOGE(LOG_ERROR, "error writing record->function");
        goto cleanup;
    }
    size = sizeof(record->result);
    if (fwrite(&record->result,
               1, size, fp) != size) {
        LOGE(LOG_ERROR, "error writing record->function");
        goto cleanup;
    }
    ret = 0;
 cleanup:
    return ret;
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
        if (record->function == CUDA_MALLOC) {
            if (cr_dump_memory(path, record) != 0) {
                LOGE(LOG_ERROR, "error dumping memory");
                goto cleanup;
            }
        }
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
    
    res = 0;
cleanup:
    return res;
}

extern void rpc_dispatch(struct svc_req *rqstp, xdrproc_t *ret_arg, xdrproc_t *ret_res, size_t *res_sz, bool_t (**ret_fun)(char *, void *, struct svc_req *));

int cr_call_record(api_record_t *record)
{
    struct svc_req rqstp;
    xdrproc_t arg;
    xdrproc_t res;
    size_t res_sz;
    bool_t retval;
    void *result;
    bool_t (*fun)(char *, void *, struct svc_req *);

    rqstp.rq_proc = record->function;
    rpc_dispatch(&rqstp, &arg, &res, &res_sz, &fun);
    result = malloc(res_sz);
	retval = (bool_t) (*fun)((char *)record->arguments, result, &rqstp);

    return (retval==1 ? 0 : 1);
}

static int cr_restore_resources(const char *path, api_record_t *record, resource_mg *rm_memory, resource_mg *rm_streams)
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
        break;
    case CUDA_STREAM_CREATE:
    case CUDA_STREAM_CREATE_WITH_FLAGS:
    case CUDA_STREAM_CREATE_WITH_PRIORITY:
        if (cr_restore_streams(record, rm_streams) != 0) {
            LOGE(LOG_ERROR, "error restoring streams");
            goto cleanup;
        }
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

int cr_restore(const char *path, resource_mg *rm_memory, resource_mg *rm_streams)
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
        LOGE(LOG_ERROR, "error while opening file");
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
        if (cr_restore_resources(path, record, rm_memory, rm_streams) != 0) {
            LOGE(LOG_ERROR, "error restoring resources");
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

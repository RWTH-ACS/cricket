#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>

#include "api-recorder.h"
#include "log.h"
#include "cpu_rpc_prot.h"


list api_records;

void api_records_free_args(void)
{
    api_record_t *record;
    for (size_t i = 0; i < api_records.length; i++) {
        if (list_at(&api_records, i, (void**)&record) != 0) {
            LOGE(LOG_ERROR, "list_at %zu returned an error.", i);
            continue;
        }
        free(record->arguments);
        record->arguments = NULL;
    }

}

void api_records_print(void)
{
    api_record_t *record;
    printf("server api records:\n");
    for (size_t i = 0; i < api_records.length; i++) {
        if (list_at(&api_records, i, (void**)&record) != 0) {
            LOGE(LOG_ERROR, "list_at returned an error.");
        }
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

}

int api_records_dump(FILE *file)
{
    api_record_t *record;
    size_t size;
    if (file == NULL) {
        LOGE(LOG_ERROR, "file is NULL");
        return 1;
    }
    if (ferror(file)) {
        LOGE(LOG_ERROR, "file descriptor is invalid");
        return 1;
    }
    for (size_t i = 0; i < api_records.length; i++) {
        if (list_at(&api_records, i, (void**)&record) != 0) {
            LOGE(LOG_ERROR, "list_at %zu returned an error.", i);
            continue;
        }
        size = sizeof(record->function);
        if (fwrite(&record->function,
                   size, 1, file) != 1) {
            LOGE(LOG_ERROR, "error writing record->function");
            return 1;
        }
        size = sizeof(record->arg_size);
        if (fwrite(&record->arg_size,
                   size, 1, file) != 1) {
            LOGE(LOG_ERROR, "error writing record->function");
            return 1;
        }
        size = record->arg_size;
        if (fwrite(record->arguments,
                   size, 1, file) != 1) {
            LOGE(LOG_ERROR, "error writing record->function");
            return 1;
        }
        size = sizeof(record->result);
        if (fwrite(&record->result,
                   size, 1, file) != 1) {
            LOGE(LOG_ERROR, "error writing record->function");
            return 1;
        }
    }
    return 0;
}

int api_records_restore(FILE *file)
{
    api_record_t *record;
    size_t size;
    int function;
    if (file == NULL) {
        LOGE(LOG_ERROR, "file is NULL");
        return 1;
    }
    if (ferror(file) || feof(file)) {
        LOGE(LOG_ERROR, "file descriptor is invalid");
        return 1;
    }
    while (!feof(file)) {
        size = sizeof(record->function);
        if (fread(&function,
                   1, size, file) != size) {
            if (feof(file)) {
                break;
            }
            LOGE(LOG_ERROR, "error reading record->function");
            return 1;
        }
        if (list_append(&api_records, (void**)&record) != 0) {
            LOGE(LOG_ERROR, "list_append returned an error.");
            continue;
        }
        record->function = function;
        size = sizeof(record->arg_size);
        if (fread(&record->arg_size,
                   1, size, file) != size) {
            LOGE(LOG_ERROR, "error reading record->function");
            return 1;
        }
        size = record->arg_size;
        if ( (record->arguments = malloc(size)) == NULL) {
            LOGE(LOG_ERROR, "malloc failed");
            return 1;
        }
        if (fread(record->arguments,
                   1, size, file) != size) {
            LOGE(LOG_ERROR, "error reading record->function");
            return 1;
        }
        size = sizeof(record->result);
        if (fread(&record->result,
                   1, size, file) != size) {
            LOGE(LOG_ERROR, "error reading record->function");
            return 1;
        }
    }
    return 0;
}

#include <stdlib.h>
#include <stdio.h>

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

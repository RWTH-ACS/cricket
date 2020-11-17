#ifndef _API_RECODER_H_
#define _API_RECODER_H_

#include <stdint.h>
#include "list.h"

#ifdef WITH_RECORDER
#define RECORD_VOID_API \
    api_record_t *record; \
    if (list_append(&api_records, (void**)&record) != 0) { \
        LOGE(LOG_ERROR, "list allocation failed."); \
    } \
    record->function = rqstp->rq_proc; \
    record->arguments = NULL;
#define RECORD_API(ARG_TYPE) \
    api_record_t *record; \
    ARG_TYPE *arguments; \
    if (list_append(&api_records, (void**)&record) != 0) { \
        LOGE(LOG_ERROR, "list allocation failed."); \
    } \
    if ( (arguments = malloc(sizeof(ARG_TYPE))) == NULL) { \
        LOGE(LOG_ERROR, "list arguments allocation failed"); \
    } \
    record->function = rqstp->rq_proc; \
    record->arguments = arguments;
#define RECORD_RESULT(TYPE, RES) \
    record->result.TYPE = RES
#define RECORD_SINGLE_ARG(ARG) \
    *arguments = ARG
#define RECORD_ARG(NUM, ARG) \
    arguments->arg##NUM = ARG
#else
#define RECORD_API(ARG_TYPE) 
#define RECORD_RESULT(TYPE, RES)
#define RECORD_ARG(NUM, ARG)
#define RECORD_SINGLE_ARG(ARG)
#endif //WITH_RECORDER


typedef struct api_record {
    unsigned int function;
    void *arguments;
    union {
        uint64_t u64;
        void* ptr;
        int integer;
    } result;
} api_record_t;
extern list api_records;


void api_records_free_args(void);
void api_records_print(void);

#endif

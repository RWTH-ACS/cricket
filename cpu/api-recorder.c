#include <stdlib.h>
#include "api-recorder.h"
#include "log.h"

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

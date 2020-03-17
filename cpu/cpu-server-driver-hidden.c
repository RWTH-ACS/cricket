#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cpu-server-driver-hidden.h"

#define EXPECT_CALL_CNT 4
#define EXPECT_0 9
#define EXPECT_1 6
#define EXPECT_2 2
#define EXPECT_3 3

static const int expect_elem_cnt[EXPECT_CALL_CNT] = {
    EXPECT_0,
    EXPECT_1,
    EXPECT_2,
    EXPECT_3,
};
static const int hidden_offset[EXPECT_CALL_CNT] = {
    0,
    EXPECT_0,
    EXPECT_0+EXPECT_1,
    EXPECT_0+EXPECT_1+EXPECT_2,
};
static const int expect_elems_total = EXPECT_0+EXPECT_1+EXPECT_2+EXPECT_3;
                                    

static void* hidden_table[EXPECT_0+EXPECT_1+EXPECT_2+EXPECT_3] = {0};

static int call_cnt = 0;

/* append a ptr table segment to the static array above
 */
void* cd_svc_hidden_add_table(void* export_table, size_t len)
{
    if (call_cnt >= EXPECT_CALL_CNT) {
        return NULL;
    }
    if ((call_cnt == 3 && expect_elem_cnt[call_cnt] != len) || 
        (call_cnt != 3 && expect_elem_cnt[call_cnt] != len-1)) {
        return NULL;
    }
    if (call_cnt == 3) {
        memcpy(hidden_table+hidden_offset[call_cnt], export_table, len*sizeof(void*));
    } else {
        //there is a length element at the beginning of some tables
        memcpy(hidden_table+hidden_offset[call_cnt],
               export_table+sizeof(void*),
               (len-1)*sizeof(void*));
    }

    call_cnt++;

    return hidden_table+hidden_offset[call_cnt-1];
}

/* get the function pointer to a specific hidden function
 */
void *cd_svc_hidden_get(size_t call, size_t index)
{
    if (call >= EXPECT_CALL_CNT || index >= expect_elem_cnt[call])
        return NULL;

    return hidden_table[hidden_offset[call]+index];
}

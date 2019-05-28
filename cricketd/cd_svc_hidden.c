#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cd_svc_hidden.h"

#define EXPECT_CALL_CNT 4
#define EXPECT_0 8
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

int cd_svc_hidden_add_table(void* export_table, size_t len)
{
    if (call_cnt == EXPECT_CALL_CNT) {
        return 0;
    }
    if ((call_cnt == 3 && expect_elem_cnt[call_cnt] != len-1) || 
        (call_cnt != 3 && expect_elem_cnt[call_cnt] != len)) {
        return 0;
    }
    if (call_cnt == 3) {
        memcpy(hidden_table+hidden_offset[call_cnt], export_table+1, len);
    } else {
        memcpy(hidden_table+hidden_offset[call_cnt], export_table, len);
    }

    call_cnt++;

    return 1;
}

void *cd_svc_hidden_get(size_t call, size_t index)
{
    if (call >= EXPECT_CALL_CNT || index >= expect_elem_cnt[index])
        return NULL;

    return hidden_table[hidden_offset[call]+index];
}

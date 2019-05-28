#include <stdio.h>
#include <stdlib.h>
#include "cd_client_hidden.h"

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
                                    

static void* hidden_table[EXPECT_0+EXPECT_1+EXPECT_2+EXPECT_3] = {
    hidden_0_0,
    hidden_get_device_ctx,
    hidden_0_2,
    hidden_0_3,
    hidden_0_4,
    hidden_get_module,
    hidden_0_6,
    hidden_0_7,
    hidden_1_0,
    hidden_1_1,
    hidden_1_2,
    hidden_1_3,
    hidden_1_4,
    hidden_1_5,
    hidden_2_0,
    hidden_2_1,
    hidden_3_0,
    hidden_3_1,
    hidden_3_2,
};
static void* map_table[EXPECT_0+EXPECT_1+EXPECT_2+EXPECT_3] = {0};

static int call_cnt = 0;

void cd_client_hidden_reset(void)
{
    call_cnt = 0;
}

int cd_client_hidden_incr(void)
{
    return ++call_cnt;
}

void* cd_client_hidden_replace(void* orig_addr, size_t index)
{
    void *ret = NULL;
    if (call_cnt >= EXPECT_CALL_CNT) {
        fprintf(stderr, "[hidden]: too many calls!\n");
        return ret;
    }
    if (index >= expect_elem_cnt[call_cnt]) {
        fprintf(stderr, "[hidden]: too many elements for call %d\n", call_cnt);
        return ret;
    }
    map_table[hidden_offset[call_cnt]+index] = orig_addr;
    ret = hidden_table[hidden_offset[call_cnt]+index];
    return ret;
}

int hidden_0_0(void* arg1) 
{
    printf("%s() -> UNIMPLEMENTED!\n", __FUNCTION__);
}


static void* ctx_map[16] = {0};
static size_t ctx_map_cnt = 0;

/* called as part of 
 * cudart::deviceMgr::enumerateDevices()
 * second parameter is likely CUdevice = int and holds the device ID
 * first parameter is likely CUcontext* = CUctx_st** and holds a 
 * device specific context. Maybe the primary context? 
 * This context is used with cuCtxSetCurrent(CUcontext) and
 * cuDevicePrimaryCtxRetain(CUcontext*)
 */
int hidden_get_device_ctx(void** cu_ctx, int cu_device)
{
    int map_index = hidden_offset[0]+1;
    int ret = 1;
    //ret = ((int(*)(void**,int))(map_table[map_index]))(cu_ctx, cu_device);
    ret = ((int(*)(void**,int))(map_table[map_index]))
                               (&ctx_map[ctx_map_cnt++], cu_device);
    //We actually want the below statement to begin at 1 because 0 will
    //be interpreted as an error.
    *cu_ctx = (void*)ctx_map_cnt;
    printf("%s(ctx=%p->%p->(CUctx_st), device=%d) = %d\n", __FUNCTION__,
           cu_ctx, *cu_ctx, cu_device, ret);
    printf("\treal_ctx: %p\n", ctx_map[ctx_map_cnt-1]);
    return ret;
}

void* cd_client_get_fake_ctx(void* real_ctx)
{
    for (size_t i=0; i <= ctx_map_cnt; ++i) {
        if (real_ctx == ctx_map[i]) {
            return (void*)(i+1);
        }
    }
    return NULL;
}

void* cd_client_get_real_ctx(void* fake_ctx)
{
    if ((size_t)fake_ctx > ctx_map_cnt) {
        return NULL;
    }
    return ctx_map[(size_t)fake_ctx-1];
}

int hidden_0_2(void* arg1) 
{
    printf("%s() -> UNIMPLEMENTED!\n", __FUNCTION__);
}

int hidden_0_3(void* arg1) 
{
    printf("%s() -> UNIMPLEMENTED!\n", __FUNCTION__);
}

int hidden_0_4(void* arg1) 
{
    printf("%s() -> UNIMPLEMENTED!\n", __FUNCTION__);
}


static void* module_map[16] = {0};
static size_t module_map_cnt = 0;

void* cd_client_get_fake_module(void* real_module)
{
    for (size_t i=0; i <= module_map_cnt; ++i) {
        if (real_module == module_map[i]) {
            return (void*)(i+1);
        }
    }
    return NULL;
}

void* cd_client_get_real_module(void* fake_module)
{
    if ((size_t)fake_module > module_map_cnt) {
        return NULL;
    }
    return module_map[(size_t)fake_module-1];
}

/* called as part of cudart::contextState::
 * loadCubin(bool*, cudart::globalModule*)
 * seems to return the current/global module
 * arg1 = *CUmodule returns the module
 * Why does this function have so many parameters?
 */
int hidden_get_module(void** cu_module, void** arg2, void* arg3, void* arg4, int arg5) 
{
    int map_index = hidden_offset[0]+5;
    int ret = 1;
    printf("pre %s(%p->%p, %p->%p, %p, %p, %d) = %d\n", __FUNCTION__, cu_module, *cu_module, arg2, *arg2, arg3, arg4, arg5, ret);
    ret = ((int(*)(void*,void*,void*,void*,int))(map_table[map_index]))(&module_map[module_map_cnt++],arg2,arg3,arg4,arg5);
    //We actually want the below statement to begin at 1 because 0 will
    //be interpreted as an error.
    *cu_module = (void*)module_map_cnt;
    printf("post %s(%p->%p->(CUmodule_st), %p->%p, %p, %p, %d) = %d\n", __FUNCTION__, cu_module, *cu_module, arg2, *arg2, arg3, arg4, arg5, ret);
    printf("\treal_module: %p\n", module_map[module_map_cnt-1]);

    return ret;
}

/* called as part of 
 * cudart::globalState::destroyModule(cudart::globalModule*) 
 */
int hidden_0_6(void* arg1) 
{
    int map_index = hidden_offset[0]+6;
    printf("%s(%p) called\n", __FUNCTION__, arg1);
    return ((int(*)(void*))(map_table[map_index]))(arg1);
}

int hidden_0_7(void* arg1) 
{
    printf("%s() -> UNIMPLEMENTED!\n", __FUNCTION__);
}

int hidden_1_0(int arg1, void* arg2)
{
    printf("%s(%d, %p) -> UNIMPLEMENTED!\n", __FUNCTION__, arg1, arg2);
}

/* called as part of
 * cudart::globalState::initializeDriverInternal()
 */
int hidden_1_1(void* arg1, void *arg2)
{
    int map_index = hidden_offset[1]+1;
    int ret = 1;
    printf("pre %s(%p->%p, %p->%p) called\n", __FUNCTION__, arg1, *(void**)arg1, arg2, *(void**)arg2);
    ret = ((int(*)(void*,void*))(map_table[map_index]))(arg1,arg2);
    printf("post %s(%p->%p->%p, %p->%p) called\n", __FUNCTION__, arg1, *(void**)arg1, **(void***)arg1, arg2, *(void**)arg2);
    *(void**)arg1 = malloc(0x54);
    int *test_ptr = (int*)((*(char**)arg1)+0x50);
    //I have not the slightest idea what this does, but the runtime tests this address
    *test_ptr = 0;
    return ret;

}

int hidden_1_2(void* arg1) 
{
    printf("%s() -> UNIMPLEMENTED!\n", __FUNCTION__);
}

/* parameter seems correct. Is called directly from cudart api functions
 * e.g., cudaMalloc. Return value is not checked at all.
 */
int hidden_1_3(void* arg1, void* arg2) 
{
    printf("%s(%p, %p) -> UNIMPLEMENTED!\n", __FUNCTION__, arg1, arg2);
}

int hidden_1_4(void* arg1) 
{
    printf("%s() -> UNIMPLEMENTED!\n", __FUNCTION__);
}

/* called as part of
 * cudart::globalState::initializeDriverInternal()
 */
int hidden_1_5(void* arg1, void* arg2) 
{
    int map_index = hidden_offset[1]+5;
    int ret = 1;
    printf("pre %s(%p->%p, %p->%p) called\n", __FUNCTION__, arg1, *(void**)arg1, arg2, *(void**)arg2);
    ret = ((int(*)(void*,void*))(map_table[map_index]))(arg1,arg2);
    printf("post %s(%p->%p->%p, %p->%p) called\n", __FUNCTION__, arg1, *(void**)arg1, **(void***)arg1, arg2, *(void**)arg2);
    *(void**)arg1 = (void*)1;
    return ret;
}


int hidden_2_0(void* arg1) 
{
    printf("%s() -> UNIMPLEMENTED!\n", __FUNCTION__);
}

/* parameter seems correct. Is called directly from cudart api functions
 * e.g., cudaMalloc. Return value is not checked at all.
 */
int hidden_2_1(void* arg1)
{
    printf("%s(%p) -> UNIMPLEMENTED!\n", __FUNCTION__, arg1);
}

/* called as part of cudart::contextStateManager::
 * initRuntimeContextState_nonreentrant(cudart::contextState**)
 * the second parameter is a NULL-terminated function pointer array
 * the third parameter is the module (CUmodule*)
 */
int hidden_3_0(int arg1, void* arg2, void* arg3)
{
    int map_index = hidden_offset[3]+0;
    int ret = 1;
    printf("pre %s(%d, %p->%p, %p->%d) called\n", __FUNCTION__, arg1, arg2, *(void**)arg2, arg3, *(void**)arg3);
    ret = ((int(*)(int,void*,void*))(map_table[map_index]))(arg1,arg2,arg3);
    printf("pre %s(%d, %p->%p, %p->%d) called\n", __FUNCTION__, arg1, arg2, *(void**)arg2, arg3, *(void**)arg3);
    return ret;
}

/* called as part of cudart::contextStateManager::
 * destroyContextState(cudart::contextState*, bool)
 */
int hidden_3_1(void* arg1, void* arg2)
{
    int map_index = hidden_offset[3]+1;
    printf("%s(%p, %p) called\n", __FUNCTION__, arg1, arg2);
    return ((int(*)(void*,void*))(map_table[map_index]))(arg1,arg2);
}

/* called as part of cudart::contextStateManager::
 * getRuntimeContextState(cudart::contextState**, bool)
 * This seems to export the Context from driver to runtime
 * The returned context is stored in the first parameter.
 * It returns *arg1=NULL as long as the context is not yet
 * initialized and during intilization.
 * The last parameter points to a NULL-terminated function pointer array
 */
int hidden_3_2(void** arg1, int arg2, void** arg3)
{
    int map_index = hidden_offset[3]+2;
    int ret = 1;
    printf("pre %s(%p, %d, %p->%p->%p)\n", __FUNCTION__, *arg1, arg2, arg3, *arg3, **(void***)arg3);
    ret = ((int(*)(void*,int,void*))(map_table[map_index]))(arg1,arg2,arg3);
    //printf("pre %s(%p, %d, %p->%p->%p)\n", __FUNCTION__, *arg1, arg2, arg3, *arg3, **(void***)arg3);
    printf("post %s(%p, %d, %p->%p->%p)\n", __FUNCTION__, *arg1, arg2, arg3, *arg3);
    if (*arg1 != NULL) {
        printf("\t->%p\n", **(void***)arg1);
    }
    return ret;
}

#include <stdio.h>
#include <stdlib.h>

#include "cpu_rpc_prot.h"

#include "cpu-common.h"
#include "cpu-client-driver-hidden.h"

#define EXPECT_CALL_CNT 4
#define EXPECT_0 9
#define EXPECT_1 7
#define EXPECT_2 3
#define EXPECT_3 3

static const int expect_elem_cnt[EXPECT_CALL_CNT] = {
    EXPECT_0,
    EXPECT_1,
    EXPECT_2,
    EXPECT_3,
};
static const int hidden_offset[EXPECT_CALL_CNT] = {
    0,
    EXPECT_0+1,
    EXPECT_0+EXPECT_1+2,
    EXPECT_0+EXPECT_1+EXPECT_2+3,
};
static const int expect_elems_total = EXPECT_0+EXPECT_1+EXPECT_2+EXPECT_3+EXPECT_CALL_CNT;
                                    
static void* hidden_table[EXPECT_0+EXPECT_1+EXPECT_2+EXPECT_3+EXPECT_CALL_CNT] = {
    (void*)(EXPECT_0*sizeof(void*)),
    hidden_0_0, //CU_ETID_CudartInterface
    hidden_get_device_ctx,
    hidden_0_2, //CU_ETID_ToolsRuntimeCallbackHooks
    hidden_0_3,
    hidden_0_4, //CU_ETID_ContextLocalStorageInterface_v0301
    hidden_get_module,
    hidden_0_6,
    hidden_0_7,
    NULL,
    (void*)(EXPECT_1*sizeof(void*)),
    hidden_1_0,
    hidden_1_1,
    hidden_1_2,
    hidden_1_3,
    hidden_1_4,
    hidden_1_5,
    NULL,
    (void*)(EXPECT_2*sizeof(void*)),
    hidden_2_0,
    hidden_2_1,
    NULL,
    hidden_3_0,
    hidden_3_1,
    hidden_3_2,
    NULL,
};

static void* orig_ptrs[EXPECT_CALL_CNT] = {0};

static int call_cnt = 0;

void cd_client_hidden_init(void *new_clnt)
{
    clnt = (CLIENT*)new_clnt;
}

void cd_client_hidden_reset(void)
{
    call_cnt = 0;
}

int cd_client_hidden_incr(void)
{
    return ++call_cnt;
}

void *cd_client_hidden_get(void *orig_ptr)
{
    //lets remember the original pointer in the server address space
    //so we can use it when we do the next RPC
    orig_ptrs[call_cnt] = orig_ptr;
    return hidden_table+hidden_offset[call_cnt];
}

/* get ptr to function ptr array in server address space from a ptr in client space
 * see cd_client_hidden_get
 */
void *cd_client_hidden_orig_ptr(void *replaced_ptr)
{
    for (int i = 0; i < EXPECT_CALL_CNT; ++i) {
        if (hidden_table+hidden_offset[i] == replaced_ptr) {
            return orig_ptrs[i];
        }
    }
    return NULL;
}

/*
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
}*/

int hidden_0_0(void* arg1) 
{
    printf("%s() -> UNIMPLEMENTED!\n", __FUNCTION__);
}


static void* ctx_map_orig[16] = {0};
static void* ctx_map_fake[16] = {0};
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
 /*   int map_index = hidden_offset[0]+1;
    int ret = 1;
    //ret = ((int(*)(void**,int))(map_table[map_index]))(cu_ctx, cu_device);
    //ret = ((int(*)(void**,int))(map_table[map_index]))
    //                           (&ctx_map[ctx_map_cnt++], cu_device);
    //We actually want the below statement to begin at 1 because 0 will
    //be interpreted as an error.
    *cu_ctx = (void*)ctx_map_cnt;
    printf("%s(ctx=%p->%p->(CUctx_st), device=%d) = %d\n", __FUNCTION__,
           cu_ctx, *cu_ctx, cu_device, ret);
    printf("\treal_ctx: %p\n", ctx_map[ctx_map_cnt-1]);
    return ret;*/
	enum clnt_stat retval;
    ptr_result result;
    printf("%s\n", __FUNCTION__);
    retval = rpc_hidden_get_device_ctx_1(cu_device, &result, clnt);
    printf("[rpc] %s = %d, result %p\n", __FUNCTION__, result.err,
                                        result.ptr_result_u.ptr);
	if (retval != RPC_SUCCESS) {
		fprintf(stderr, "[rpc] %s failed.", __FUNCTION__);
        return 1;
	}
    *cu_ctx = (void*)result.ptr_result_u.ptr;
    return result.err;
}

void* cd_client_get_fake_ctx(void* real_ctx)
{
    size_t i;
    uint64_t *ctx_elem = malloc(sizeof(uint64_t));
    for (i=0; i <= ctx_map_cnt; ++i) {
        if (real_ctx == ctx_map_orig[i]) {
            return ctx_map_fake[i];
        }
    }
    if (i == 16) return NULL;
    ctx_map_fake[i] = calloc(1,0x58);
    ((uint64_t*)ctx_map_fake[i])[6] = 0x6;
    // cudart::contextState::getEntryFunction dereferences ctx+0x40.
    // This is called during kernel startup
    ((uint64_t**)ctx_map_fake[i])[8] = ctx_elem;
    *ctx_elem = (uint64_t)&((uint64_t**)ctx_map_fake[i])[9];
    ((uint64_t**)ctx_map_fake[i])[10] = (uint64_t*)0x403ca9;
    ctx_map_orig[i] = real_ctx;
    ctx_map_cnt++;
    return ctx_map_fake[i];
}

void* cd_client_get_real_ctx(void* fake_ctx)
{
    size_t i;
    for (i=0; i <= ctx_map_cnt; ++i) {
        if (fake_ctx == ctx_map_fake[i]) {
            return ctx_map_orig[i];
        }
    }
    return NULL;
}

//TODO: use!
void cd_client_free_ctx_map()
{
    size_t i;
    for (i=0; i <= ctx_map_cnt; ++i) {
        free(((uint64_t**)ctx_map_fake[i])[8]);
        free(ctx_map_fake[i]);
    }
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
 * arg2 is a pointer to a pointer to the symbol fatDeviceText in 
 * the program binary (use elfread!) that, presumably,
 * contains the device code.
 * This function seems to be similar to cuModuleLoadDataEx (at least the 
 * signature is almost identical and the functions seems to be the same as
 * well. However, this is a distinct function with different disassembly)
 */
int hidden_get_module(void** cu_module, void** arg2, void* arg3, void* arg4, int arg5) 
{
	enum clnt_stat retval;
    ptr_result result;
    
    //printf("pre %s(%p->%p, %p->%p, %p, %p, %d)\n", __FUNCTION__, cu_module, *cu_module, arg2, *arg2, arg3, arg4, arg5);

    retval = rpc_hidden_get_module_1((uint64_t)*arg2, (uint64_t)arg3,
                                     (uint64_t)arg4, arg5, &result, clnt);
    *cu_module = (void*)result.ptr_result_u.ptr;
    printf("[rpc] %s(%p->%p->(CUmodule_st), %p->%p, %p, %p, %d) = %d\n", __FUNCTION__, cu_module, *cu_module, arg2, *arg2, arg3, arg4, arg5, result.err);

    return result.err;
}

/* called as part of 
 * cudart::globalState::destroyModule(cudart::globalModule*) 
 */
int hidden_0_6(void* arg1) 
{
    int map_index = hidden_offset[0]+6;
    printf("%s(%p) called\n", __FUNCTION__, arg1);
    //return ((int(*)(void*))(map_table[map_index]))(arg1);
    return 1;
}

int hidden_0_7(void* arg1) 
{
    printf("%s() -> UNIMPLEMENTED!\n", __FUNCTION__);
}

/* Used
 */
int hidden_1_0(int arg1, void* arg2)
{
    printf("%s(%d, %p) -> UNIMPLEMENTED!\n", __FUNCTION__, arg1, arg2);
}

/* called as part of
 * cudart::globalState::initializeDriverInternal()
 */
int hidden_1_1(void* arg1, void *arg2)
{
    //printf("pre %s(%p->%p, %p->%p) called\n", __FUNCTION__, arg1, *(void**)arg1, arg2, *(void**)arg2);

	enum clnt_stat retval;
    ptr_result result;
    retval = rpc_hidden_1_1_1(&result, clnt);
    printf("[rpc] %s = %d\n", __FUNCTION__, result.err);
	if (retval != RPC_SUCCESS) {
		fprintf(stderr, "[rpc] %s failed.", __FUNCTION__);
        return 1;
	}

    //printf("post %s(%p->%p, %p->%p) called\n", __FUNCTION__, arg1, *(void**)arg1, arg2, *(void**)arg2);
    *(void**)arg2 = (void*)result.ptr_result_u.ptr;
    *(void**)arg1 = malloc(0x54);
    int *test_ptr = (int*)((*(char**)arg1)+0x50);
    //I have not the slightest idea what this does, but the runtime tests this address
    *test_ptr = 0;
    return result.err;
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
    printf("%s(%p, %p->%p) -> UNIMPLEMENTED!\n", __FUNCTION__, arg1, arg2, *(void**)arg2);
}

int hidden_1_4(void* arg1) 
{
    printf("%s() -> UNIMPLEMENTED!\n", __FUNCTION__);
}

/* called as part of
 * cudart::globalState::initializeDriverInternal()
 * I have no clue what this does and whether the below is correct.
 * The calling function seems to do not much else than check that the pointers are
 * non-NULL (better verify this before assuming this statement is correct).
 */
int hidden_1_5(void* arg1, void* arg2) 
{
	enum clnt_stat retval;
    ptr_result result;
    //printf("pre %s(%p->%p, %p->%p) called\n", __FUNCTION__, arg1, *(void**)arg1, arg2, *(void**)arg2);
    retval = rpc_hidden_1_5_1(&result, clnt);
    printf("[rpc] %s = %d\n", __FUNCTION__, result.err);
	if (retval != RPC_SUCCESS) {
		fprintf(stderr, "[rpc] %s failed.", __FUNCTION__);
        return 1;
	}

    //printf("post %s(%p->%p->%p, %p->%p) called\n", __FUNCTION__, arg1, *(void**)arg1, **(void***)arg1, arg2, *(void**)arg2);
    *(void**)arg2 = (void*)result.ptr_result_u.ptr;
    *(void**)arg1 = (void*)1;
    return result.err;
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
    printf("%s(%p->%p) -> UNIMPLEMENTED!\n", __FUNCTION__, arg1, *(void**)arg1);
}

/* called as part of cudart::contextStateManager::
 * initRuntimeContextState_nonreentrant(cudart::contextState**)
 * the second parameter is a NULL-terminated function pointer array
 * the third parameter is the module (CUmodule*)
 */
int hidden_3_0(int arg1, void** arg2, void** arg3)
{
	enum clnt_stat retval;
    int result;
    void *arg2_orig = cd_client_hidden_orig_ptr(*arg2);
    //printf("pre %s(%p, %d, %p->%p->%p)\n", __FUNCTION__, *arg1, arg2, arg3, *arg3, **(void***)arg3);
    if (arg2_orig == NULL) {
        fprintf(stderr, "[rpc] %s failed to retrieve original ptr table\n", __FUNCTION__);
        return 1;
    }
    retval = rpc_hidden_3_0_1(arg1, (uint64_t)arg2_orig, (uint64_t)*arg3,
                              &result, clnt);
    printf("[rpc] %s(%d, %p->%p, %p->%p = %d\n", __FUNCTION__,
           arg1, arg2, arg2_orig, arg3, *arg3, result);
	if (retval != RPC_SUCCESS) {
		fprintf(stderr, "[rpc] %s failed.\n", __FUNCTION__);
        return 1;
	}
    return result;
}

/* called as part of cudart::contextStateManager::
 * destroyContextState(cudart::contextState*, bool)
 */
int hidden_3_1(void* arg1, void* arg2)
{
    int map_index = hidden_offset[3]+1;
    printf("%s(%p, %p) called\n", __FUNCTION__, arg1, arg2);
    //return ((int(*)(void*,void*))(map_table[map_index]))(arg1,arg2);
    return 1;
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
	enum clnt_stat retval;
    mem_result result;
    result.mem_result_u.data.mem_data_val = malloc(0x58);
    void *arg3_orig = cd_client_hidden_orig_ptr(*arg3);
    //printf("\tppre %s(%p->%p, %d, %p->%p->%p)\n", __FUNCTION__, arg1, *arg1, arg2, arg3, *arg3, **(void***)arg3);
    //printf("\tfaked arg3: %p\n", arg3_orig);
    if (arg3_orig == NULL) {
        fprintf(stderr, "[rpc] %s failed to retrieve original ptr table\n", __FUNCTION__);
        return 1;
    }
    retval = rpc_hidden_3_2_1(arg2, (uint64_t)arg3_orig, &result, clnt);
    printf("[rpc] %s = %d, result = %p\n", __FUNCTION__, result.err,
           (void*)result.mem_result_u.data.mem_data_val);
	if (retval != RPC_SUCCESS) {
		fprintf(stderr, "[rpc] %s failed.\n", __FUNCTION__);
        return 1;
	}
    *arg1 = result.mem_result_u.data.mem_data_val;
    //*arg1 = cd_client_get_fake_ctx((void*)result.ptr_result_u.ptr);
    //printf("\tfaked result: %p\n", *arg1);

    if (*arg1 != 0)
        printf("\t%p, @0x30: %p, @0x40: %p\n", *arg1, (*(void***)arg1)[6], (*(void***)arg1)[8]);
    //*arg1 = (void*)result.ptr_result_u.ptr;
    //printf("\tppost %s(%p->%p, %d, %p->%p->%p)\n", __FUNCTION__, arg1, *arg1, arg2, arg3, *arg3, **(void***)arg3);
    return result.err;
}

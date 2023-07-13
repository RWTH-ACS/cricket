#include <stdlib.h>
#include <stdio.h>

#include <cuda.h>

#include "cpu_rpc_prot.h"
#include "cpu-server-driver-hidden.h"
#include "cpu-common.h"
#include "cpu-utils.h"
#include "log.h"
#include "resource-mg.h"
#define WITH_RECORDER
#include "api-recorder.h"
#include "gsched.h"

int server_driver_init(int restore)
{
    #ifdef WITH_IB
    #endif //WITH_IB
   
    int ret = 0;
    if (!restore) {
        // we cannot bypass the resource manager for functions and modules
        // because CUfunctions and modules are at different locations on server and client
        ret &= resource_mg_init(&rm_modules, 0);
        ret &= resource_mg_init(&rm_functions, 0);
        ret &= resource_mg_init(&rm_globals, 0);
    } else {
        ret &= resource_mg_init(&rm_modules, 0);
        ret &= resource_mg_init(&rm_functions, 0);
        ret &= resource_mg_init(&rm_globals, 0);
        //ret &= server_driver_restore("ckp");
    }
    return ret;
}

#include <cuda_runtime_api.h>

// Does not support checkpoint/restart yet
bool_t rpc_elf_load_1_svc(mem_data elf, ptr module_key, int *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "rpc_elf_load(elf: %p, len: %#x, module_key: %#x)", elf.mem_data_val, elf.mem_data_len, module_key);
    CUresult res;
    CUmodule module = NULL;
    
    if ((res = cuModuleLoadData(&module, elf.mem_data_val)) != CUDA_SUCCESS) {
        LOGE(LOG_ERROR, "cuModuleLoadData failed: %d", res);
        *result = res;
        return 1;
    }

    // We add our module using module_key as key. This means a fatbinaryHandle on the client is translated
    // to a CUmodule on the server.
    if ((res = resource_mg_add_sorted(&rm_modules, (void*)module_key, (void*)module)) != CUDA_SUCCESS) {
        LOGE(LOG_ERROR, "resource_mg_create failed: %d", res);
        *result = res;
        return 1;
    }

    LOGE(LOG_DEBUG, "->module: %p", module);
    *result = 0;
    return 1;
}

// Does not support checkpoint/restart yet
// TODO: We should also remove associated function handles
bool_t rpc_elf_unload_1_svc(ptr elf_handle, int *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "rpc_elf_unload(elf_handle: %p)", elf_handle);
    CUmodule module = NULL;
    CUresult res;
    
    if ((module = (CUmodule)resource_mg_get(&rm_modules, (void*)elf_handle)) == NULL) {
        LOG(LOG_ERROR, "resource_mg_get failed");
        *result = -1;
        return 1;
    }

    LOGE(LOG_DEBUG,"module: %p", module);

    // if ((res = resource_mg_remove(&rm_modules, (void*)elf_handle)) != CUDA_SUCCESS) {
    //     LOG(LOG_ERROR, "resource_mg_create failed: %d", res);
    //     result->err = res;
    //     return 1;
    // }

    if ((res = cuModuleUnload(module)) != CUDA_SUCCESS) {
        const char *errstr;
        cuGetErrorString(res, &errstr);
        LOG(LOG_ERROR, "cuModuleUnload failed: %s (%d)", errstr, res);
        *result = res;
        return 1;
    }

    *result = 0;
    return 1;
}

// Does not support checkpoint/restart yet
bool_t rpc_register_function_1_svc(ptr fatCubinHandle, ptr hostFun, char* deviceFun,
                            char* deviceName, int thread_limit, ptr_result *result, struct svc_req *rqstp)
{
    void *module = NULL;
    RECORD_API(rpc_register_function_1_argument);
    RECORD_ARG(1, fatCubinHandle);
    RECORD_ARG(2, hostFun);
    RECORD_ARG(3, deviceFun);
    RECORD_ARG(4, deviceName);
    RECORD_ARG(5, thread_limit);
    LOG(LOG_DEBUG, "rpc_register_function(fatCubinHandle: %p, hostFun: %p, deviceFun: %s, deviceName: %s, thread_limit: %d)",
        fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit);
    GSCHED_RETAIN;
    //resource_mg_print(&rm_modules);
    if ((module = resource_mg_get(&rm_modules, (void*)fatCubinHandle)) == (void*)fatCubinHandle) {
        LOGE(LOG_ERROR, "%p not found in resource manager - we cannot call a function from an unknown module.", fatCubinHandle);
        result->err = -1;
        return 1;
    }
    result->err = cuModuleGetFunction((CUfunction*)&result->ptr_result_u.ptr,
                    module,
                    deviceName);
    if (resource_mg_add_sorted(&rm_functions, (void*)hostFun, (void*)result->ptr_result_u.ptr) != 0) {
        LOGE(LOG_ERROR, "error in resource manager");
    }
    GSCHED_RELEASE;
    RECORD_RESULT(ptr_result_u, *result);
    return 1;
}

// Does not support checkpoint/restart yet
bool_t rpc_register_var_1_svc(ptr fatCubinHandle, ptr hostVar, ptr deviceAddress, char *deviceName, int ext, size_t size,
                        int constant, int global, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_register_var_1_argument);
    RECORD_ARG(1, fatCubinHandle);
    RECORD_ARG(2, hostVar);
    RECORD_ARG(3, deviceAddress);
    RECORD_ARG(4, deviceName);
    RECORD_ARG(5, ext);
    RECORD_ARG(6, size);
    RECORD_ARG(7, constant);
    RECORD_ARG(8, global);
    
    LOG(LOG_DEBUG, "rpc_register_var(fatCubinHandle: %p, hostVar: %p, deviceAddress: %p, deviceName: %s, "
                   "ext: %d, size: %d, constant: %d, global: %d)",
                   fatCubinHandle, hostVar, deviceAddress, deviceName, ext, size, constant, global);
    
    CUdeviceptr dptr = 0;
    size_t d_size = 0;
    CUresult res;
    void *module = NULL;
    GSCHED_RETAIN;
    if ((module = resource_mg_get(&rm_modules, (void*)fatCubinHandle)) == (void*)fatCubinHandle) {
        LOGE(LOG_ERROR, "%p not found in resource manager - we cannot call a function from an unknown module.", fatCubinHandle);
        *result = -1;
        return 1;
    }
    if ((res = cuModuleGetGlobal(&dptr, &d_size, module, deviceName)) != CUDA_SUCCESS) {
        LOGE(LOG_ERROR, "cuModuleGetGlobal failed: %d", res);
        *result = 1;
        return 1;
    }
    if (resource_mg_add_sorted(&rm_globals, (void*)hostVar, (void*)dptr) != 0) {
        LOGE(LOG_ERROR, "error in resource manager");
        *result = 1;
    } else {
        *result = 0;
    }
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

int server_driver_deinit(void)
{
    resource_mg_free(&rm_modules);
    resource_mg_free(&rm_functions);
    return 0;
}

bool_t rpc_cudevicegetcount_1_svc(int_result *result, struct svc_req *rqstp)
{
    LOG(LOG_DEBUG, "%s", __FUNCTION__);
    result->err = cuDeviceGetCount(&result->int_result_u.data);
    return 1;
}

bool_t rpc_cuinit_1_svc(int argp, int *result, struct svc_req *rqstp)
{
    LOG(LOG_DEBUG, "%s", __FUNCTION__);
    *result = cuInit(argp);
    return 1;
}

bool_t rpc_cudrivergetversion_1_svc(int_result *result, struct svc_req *rqstp)
{
    LOG(LOG_DEBUG, "%s", __FUNCTION__);
    result->err = cuDriverGetVersion(&result->int_result_u.data);
    return 1;
}

bool_t rpc_cudeviceget_1_svc(int ordinal, int_result *result, struct svc_req *rqstp)
{
    LOG(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    result->err = cuDeviceGet(&result->int_result_u.data, ordinal);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudevicegetname_1_svc(int dev, str_result *result, struct svc_req *rqstp)
{
    result->str_result_u.str = malloc(128);
    LOG(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    result->err = cuDeviceGetName(result->str_result_u.str, 128, dev);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudevicetotalmem_1_svc(int dev, u64_result *result, struct svc_req *rqstp)
{
    LOG(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    result->err = cuDeviceTotalMem(&result->u64_result_u.u64, dev);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudevicegetattribute_1_svc(int attribute, int dev, int_result *result, struct svc_req *rqstp)
{
    LOG(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    result->err = cuDeviceGetAttribute(&result->int_result_u.data, attribute, dev);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudevicegetuuid_1_svc(int dev, str_result *result, struct svc_req *rqstp)
{
    CUuuid uuid;
    LOG(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    result->err = cuDeviceGetUuid(&uuid, dev);
    GSCHED_RELEASE;
    if (result->err == 0) {
        memcpy(result->str_result_u.str, uuid.bytes, 16);
    }
    return 1;
}

bool_t rpc_cuctxgetcurrent_1_svc(ptr_result *result, struct svc_req *rqstp)
{
    LOG(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    result->err = cuCtxGetCurrent((struct CUctx_st**)&result->ptr_result_u.ptr);
    GSCHED_RELEASE;
    if ((void*)result->ptr_result_u.ptr != NULL) {
        unsigned int version = 0;
        cuCtxGetApiVersion((CUcontext)result->ptr_result_u.ptr, &version);
        LOG(LOG_DEBUG, "ctxapi version: %d", version);
    }
    return 1;
}

//TODO: Calling this might break things within the scheduler.
bool_t rpc_cuctxsetcurrent_1_svc(uint64_t ptr, int *result, struct svc_req *rqstp)
{
    LOG(LOG_DEBUG, "%s", __FUNCTION__);
    *result = cuCtxSetCurrent((struct CUctx_st*)ptr);
    return 1;
}

//TODO: Calling this might break things within the scheduler.
bool_t rpc_cudeviceprimaryctxretain_1_svc(int dev, ptr_result *result,
                                          struct svc_req *rqstp)
{
    LOG(LOG_DEBUG, "%s", __FUNCTION__);
    result->err = cuDevicePrimaryCtxRetain((struct CUctx_st**)&result->ptr_result_u.ptr, dev);
    return 1;
}

bool_t rpc_cumodulegetfunction_1_svc(uint64_t module, char *name, ptr_result *result,
                                     struct svc_req *rqstp)
{
    RECORD_API(rpc_cumodulegetfunction_1_argument);
    RECORD_ARG(1, module);
    RECORD_ARG(2, name);
    LOG(LOG_DEBUG, "(fd:%d) %s(%s)", rqstp->rq_xprt->xp_fd, __FUNCTION__, name);
    GSCHED_RETAIN;
    result->err = cuModuleGetFunction((CUfunction*)&result->ptr_result_u.ptr,
                    resource_mg_get(&rm_streams, (void*)module),
                    name);
    GSCHED_RELEASE;
    if (resource_mg_create(&rm_functions, (void*)result->ptr_result_u.ptr) != 0) {
        LOGE(LOG_ERROR, "error in resource manager");
    }
    RECORD_RESULT(ptr_result_u, *result);
    return 1;
}

bool_t rpc_cumoduleloaddata_1_svc(mem_data mem, ptr_result *result,
                                     struct svc_req *rqstp)
{
    RECORD_API(mem_data);
    RECORD_SINGLE_ARG(mem);
    LOG(LOG_DEBUG, "%s(%p, %#0zx)", __FUNCTION__, mem.mem_data_val, mem.mem_data_len);
    GSCHED_RETAIN;
    result->err = cuModuleLoadData((CUmodule*)&result->ptr_result_u.ptr, mem.mem_data_val);
    GSCHED_RELEASE;
    if (resource_mg_create(&rm_modules, (void*)result->ptr_result_u.ptr) != 0) {
        LOGE(LOG_ERROR, "error in resource manager");
    }
    if (result->err != 0) {
        char *err_str = NULL;
        cuGetErrorName(result->err, &err_str);
        LOGE(LOG_DEBUG, "cuModuleLoadData result: %s", err_str);
    }
    RECORD_RESULT(ptr_result_u, *result);
    return 1;
}
bool_t rpc_cumoduleload_1_svc(char* path, ptr_result *result,
                                     struct svc_req *rqstp)
{
    RECORD_API(char*);
    RECORD_SINGLE_ARG(path);
    LOG(LOG_DEBUG, "%s(%s)", __FUNCTION__, path);
    GSCHED_RETAIN;
    result->err = cuModuleLoad((CUmodule*)&result->ptr_result_u.ptr, path);
    GSCHED_RELEASE;
    if (resource_mg_create(&rm_modules, (void*)result->ptr_result_u.ptr) != 0) {
        LOGE(LOG_ERROR, "error in resource manager");
    }
    if (result->err != 0) {
        char *err_str = NULL;
        cuGetErrorName(result->err, &err_str);
        LOGE(LOG_DEBUG, "cuModuleLoad result: %s", err_str);
    }
    RECORD_RESULT(ptr_result_u, *result);
    return 1;
}

bool_t rpc_cumoduleunload_1_svc(ptr module, int *result,
                                     struct svc_req *rqstp)
{
    RECORD_API(ptr);
    RECORD_SINGLE_ARG(module);
    LOG(LOG_DEBUG, "%s(%p)", __FUNCTION__, (void*)module);
    GSCHED_RETAIN;
    *result = cuModuleUnload(resource_mg_get(&rm_modules, (void*)module));
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cugeterrorstring_1_svc(int err, str_result *result,
                                     struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    const char* err_str = NULL;
    result->err = cuGetErrorString(err, &err_str);

    if ((result->str_result_u.str = malloc(128)) == NULL ||
        (strncpy(result->str_result_u.str, err_str, 128) == NULL)) {
        LOGE(LOG_ERROR, "error copying string");
    }

    return 1;
}

bool_t rpc_cudeviceprimaryctxgetstate_1_svc(int dev, dint_result *result,
                                      struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s(%d)", __FUNCTION__, dev);
    GSCHED_RETAIN;
    result->err = cuDevicePrimaryCtxGetState(dev, &(result->dint_result_u.data.i1),
                                            &(result->dint_result_u.data.i2));
    LOGE(LOG_DEBUG, "state: %d, flags: %d", result->dint_result_u.data.i1,
                                           result->dint_result_u.data.i2);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudevicegetproperties_1_svc(int dev, mem_result *result,
                                       struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s(%d)", __FUNCTION__, dev);
    GSCHED_RETAIN;
    if ((result->mem_result_u.data.mem_data_val = malloc(sizeof(CUdevprop))) == NULL) {
        result->err = CUDA_ERROR_OUT_OF_MEMORY;
    }
    result->mem_result_u.data.mem_data_len = sizeof(CUdevprop);
    result->err = cuDeviceGetProperties((CUdevprop*)result->mem_result_u.data.mem_data_val, dev);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudevicecomputecapability_1_svc(int dev, dint_result *result,
                                           struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s(%d)", __FUNCTION__, dev);
    GSCHED_RETAIN;
    result->err = cuDeviceComputeCapability(&(result->dint_result_u.data.i1),
                                            &(result->dint_result_u.data.i2),
                                            dev);
    GSCHED_RELEASE;
    return 1;
}

/*
bool_t rpc_cugetexporttable_1_svc(char *rpc_uuid, ptr_result *result,
                                  struct svc_req *rqstp)
{
    void *exportTable = NULL;
    size_t tablesize = 0;
    CUuuid uuid;
    LOG(LOG_DEBUG, printf("%s\n", __FUNCTION__);
    if (rpc_uuid == NULL)
        return 0;

    memcpy(uuid.bytes, rpc_uuid, 16);
    if ((result->err = cuGetExportTable((const void**)&exportTable,
                                        (const CUuuid*)&uuid) != 0)) {
        return 1;
    }
    if (((uint32_t*)exportTable)[1] > 0) {
        tablesize = 8;
        for (int i=1; i<8; ++i) {
            if (((void**)exportTable)[i] == NULL) {
                tablesize = i;
                break;
            }
        }
    } else {
        tablesize = *((uint64_t*)exportTable)/8;
    }
    printf("\ttablesize = %lu\n", tablesize);
    printf("\tpost %p->%p\n", exportTable, *(void**)exportTable);

    if (!(uint64_t)cd_svc_hidden_add_table(exportTable, tablesize)) {
        fprintf(stderr, "\tfailed to add table!\n");
        return 0;
    }
    result->ptr_result_u.ptr = (uint64_t)exportTable;

    return 1;
}*/

bool_t rpc_cumemalloc_1_svc(uint64_t size, ptr_result *result,
                                     struct svc_req *rqstp)
{
    LOG(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    result->err = cuMemAlloc_v2((CUdeviceptr*)&result->ptr_result_u.ptr, (size_t)size);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cuctxgetdevice_1_svc(int_result *result, struct svc_req *rqstp)
{
    LOG(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    result->err = cuCtxGetDevice((CUdevice*)&result->int_result_u.data);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cumemcpyhtod_1_svc(uint64_t dptr, mem_data hptr, int *result,
                                     struct svc_req *rqstp)
{
    LOG(LOG_DEBUG, "%s(%p,%p,%d)", __FUNCTION__, dptr, hptr.mem_data_val, hptr.mem_data_len);
    GSCHED_RETAIN;
    *result = cuMemcpyHtoD_v2((CUdeviceptr)dptr, hptr.mem_data_val,
                              hptr.mem_data_len);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_culaunchkernel_1_svc(uint64_t f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, uint64_t hStream, mem_data args, int* result, struct svc_req *rqstp)
{
    void **cuda_args;
    uint16_t *arg_offsets;
    size_t param_num;
    if (args.mem_data_val == NULL) {
        LOGE(LOG_ERROR, "param.mem_data_val is NULL");
        *result = CUDA_ERROR_INVALID_VALUE;
        return 1;
    }
    if (args.mem_data_len < sizeof(size_t)) {
        LOGE(LOG_ERROR, "param.mem_data_len is too small");
        *result = CUDA_ERROR_INVALID_VALUE;
        return 1;
    }
    param_num = *((size_t*)args.mem_data_val);

    if (args.mem_data_len < sizeof(size_t)+sizeof(uint16_t)*param_num) {
        LOGE(LOG_ERROR, "param.mem_data_len is too small");
        *result = CUDA_ERROR_INVALID_VALUE;
        return 1;
    }

    arg_offsets = (uint16_t*)(args.mem_data_val+sizeof(size_t));
    cuda_args = malloc(param_num*sizeof(void*));
    for (size_t i = 0; i < param_num; ++i) {
        cuda_args[i] = args.mem_data_val+sizeof(size_t)+param_num*sizeof(uint16_t)+arg_offsets[i];
        *(void**)cuda_args[i] = resource_mg_get(&rm_memory, *(void**)cuda_args[i]);
        LOGE(LOG_DEBUG, "arg: %p (%d)", *(void**)cuda_args[i], *(int*)cuda_args[i]);
    }

    LOGE(LOG_DEBUG, "cuLaunchKernel(func=%p->%p, gridDim=[%d,%d,%d], blockDim=[%d,%d,%d], args=%p, sharedMem=%d, stream=%p)", f, resource_mg_get(&rm_functions, (void*)f), gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, cuda_args, sharedMemBytes, (void*)hStream);

    GSCHED_RETAIN;
    *result = cuLaunchKernel((CUfunction)resource_mg_get(&rm_functions, (void*)f),
                              gridDimX, gridDimY, gridDimZ,
                              blockDimX, blockDimY, blockDimZ,
                              sharedMemBytes,
                              (CUstream)hStream,
                              cuda_args, NULL);
    GSCHED_RELEASE;

    free(cuda_args);
    return 1;

}

bool_t rpc_cudevicegetp2pattribute_1_svc(int attrib, ptr srcDevice, ptr dstDevice, int_result *result, struct svc_req *rqstp)
{
    LOG(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    result->err = cuDeviceGetP2PAttribute(&result->int_result_u.data, (CUdevice_P2PAttribute)attrib, (CUdevice)srcDevice, (CUdevice)dstDevice);
    GSCHED_RELEASE;
    return 1;
}

/* ################## START OF HIDDEN FUNCTIONS IMPL ######################## */

/*
bool_t rpc_hidden_get_device_ctx_1_svc(int dev, ptr_result *result,
                                     struct svc_req *rqstp)
{
    printf("%s\n", __FUNCTION__);

    result->err = ((int(*)(void**,int))(cd_svc_hidden_get(0,1)))
                                        ((void**)&result->ptr_result_u.ptr, dev);
    return 1;
}
*/
/* This function loads the module (the device code)
 * It could be replaced by cuModuleLoad which loads a module via a path
 * string. However, this seems to only work for cubin files that were
 * compiled using "nvcc -cubin" or extracted from a fatbinary using
 * "cuobjdump -xelf". When this returns 200 or 300 the cubin may be
 * compiled for the wrong compute capabilities (e.g. Pascal needs sm_61 and
 * Turing needs sm_75).
 */
/*bool_t rpc_hidden_get_module_1_svc(uint64_t arg2, uint64_t arg3,
                                   uint64_t arg4, int arg5,
                                   ptr_result *result, struct svc_req *rqstp)
{
    printf("%s\n", __FUNCTION__);
    //TODO: make a parameter. Probably must be globally stored somehow
    //      thread-safety may be an issue
    //char str[] = "/home/eiling/projects/cricket/tests/test_api.1.sm_61.cubin";
    //result->err = cuModuleLoad((CUmodule*)&result->ptr_result_u.ptr, str);

    result->err = ((int(*)(void**,void*,uint64_t,uint64_t,int))
                   (cd_svc_hidden_get(0,5)))
                   ((void**)&result->ptr_result_u.ptr, &arg2, arg3,
                    arg4, arg5);
    return 1;
}

bool_t rpc_hidden_1_1_1_svc(ptr_result *result,
                            struct svc_req *rqstp)
{
    printf("%s\n", __FUNCTION__);
    void *l_arg1 = NULL;

    ((int(*)(void**, void**))(cd_svc_hidden_get(1,1)))
                             (&l_arg1, (void**)&result->ptr_result_u.ptr);
    result->err = 0;
    return 1;
}

bool_t rpc_hidden_1_3_1_svc(uint64_t arg1, uint64_t arg2, void* unused,
                            struct svc_req *rqstp)
{
    printf("%s\n", __FUNCTION__);

    ((void(*)(uint64_t, uint64_t))(cd_svc_hidden_get(1,3)))
                             (arg1, arg2);
    return 1;
}

bool_t rpc_hidden_1_5_1_svc(ptr_result *result,
                            struct svc_req *rqstp)
{
    printf("%s\n", __FUNCTION__);
    void *l_arg1 = NULL;

    ((int(*)(void**, void**))(cd_svc_hidden_get(1,5)))
                             (&l_arg1, (void**)&result->ptr_result_u.ptr);
    result->err = 0;
    return 1;
}

bool_t rpc_hidden_2_1_1_svc(uint64_t arg1, void* unused,
                            struct svc_req *rqstp)
{
    printf("%s\n", __FUNCTION__);

    ((void(*)(uint64_t))(cd_svc_hidden_get(2,1)))
                             (arg1);
    return 1;
}

bool_t rpc_hidden_3_0_1_svc(int arg1, uint64_t arg2, uint64_t arg3,
                            int *result, struct svc_req *rqstp)
{
    printf("%s(%d, %p, %p)\n", __FUNCTION__, arg1, arg2, arg3);
    void *fptr = cd_svc_hidden_get(3,0);
    *result = ((int(*)(int, void*, void*))(fptr))
                      (arg1, &arg2, &arg3);
    return 1;
}

bool_t rpc_hidden_3_2_1_svc(int arg2, uint64_t arg3, mem_result *result,
                            struct svc_req *rqstp)
{
    result->mem_result_u.data.mem_data_val = NULL;
    result->mem_result_u.data.mem_data_len = 0x58;
    printf("%s(%d, %p)\n", __FUNCTION__, arg2, arg3);
    //printf("\tppre %s(nh->%p, %d, nh->%p->%p)\n", __FUNCTION__, result->ptr_result_u.ptr, arg2, (void*)arg3, *(void**)arg3);
    void *fptr = cd_svc_hidden_get(3,2);
    result->err = ((int(*)(void**, int, void*))(fptr))
                             ((void**)&result->mem_result_u.data.mem_data_val, arg2, &arg3);
    void **res = ((void**)result->mem_result_u.data.mem_data_val);
    if (res != 0)
        printf("\t%p, @0x30: %p, @0x40: %p->%p\n", res, res[6], res[8], *(void**)res[8]);
    //printf("\tppost %s(nh->%p, %d, nh->%p->%p)\n", __FUNCTION__, result->ptr_result_u.ptr, arg2, (void*)arg3, *(void**)arg3);
    //printf("\terr: %d, result: %p\n", result->err, result->ptr_result_u.ptr);
    return 1;
}
*/


#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cudnn.h>

#include "cpu_rpc_prot.h"
#include "cpu-common.h"
#include "cpu-utils.h"
#include "log.h"
#include "resource-mg.h"
#include "gsched.h"

#define WITH_RECORDER
#include "api-recorder.h"

#include "cpu-server-cudnn.h"



int server_cudnn_init(int bypass, resource_mg *memory)
{
    int ret = 0;
    ret &= resource_mg_init(&rm_cudnn, bypass);
    return ret;
}

int server_cudnn_deinit(void)
{
    resource_mg_free(&rm_cudnn);
    return 0;

}

bool_t rpc_cudnngetversion_1_svc(size_t *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    *result = cudnnGetVersion();
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnngetmaxdeviceversion_1_svc(size_t *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    *result = cudnnGetMaxDeviceVersion();
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnngetcudartversion_1_svc(size_t *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    *result = cudnnGetCudartVersion();
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnngeterrorstring_1_svc(int status, char **result, struct svc_req *rqstp)
{
    const char* str;
    *result = malloc(128);
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    str = cudnnGetErrorString((cudnnStatus_t)status);
    strncpy(*result, str, 128);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnnqueryruntimeerror_1_svc(ptr handle, int mode, int_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    cudnnRuntimeTag_t *tag;

    GSCHED_RETAIN;
    result->err = cudnnQueryRuntimeError((cudnnHandle_t)handle, (cudnnStatus_t*)&result->int_result_u.data, (cudnnErrQueryMode_t)mode, tag);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnngetproperty_1_svc(int type, int_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    result->err = cudnnGetProperty((libraryPropertyType)type, &result->int_result_u.data); 
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnncreate_1_svc(ptr_result *result, struct svc_req *rqstp)
{
    RECORD_VOID_API;
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    result->err = cudnnCreate((cudnnHandle_t*)result->ptr_result_u.ptr);
    GSCHED_RELEASE;
    RECORD_RESULT(ptr_result_u, *result);
    return 1;
}

bool_t rpc_cudnndestroy_1_svc(ptr handle, int *result, struct svc_req *rqstp)
{
    RECORD_API(ptr);
    RECORD_SINGLE_ARG(handle);
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    *result = cudnnDestroy((cudnnHandle_t)handle);
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnnsetstream_1_svc(ptr handle, ptr streamId, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnsetstream_1_argument);
    RECORD_NARG(handle);
    RECORD_NARG(streamId);
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    *result = cudnnSetStream((cudnnHandle_t)handle, (cudaStream_t)streamId);
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnngetstream_1_svc(ptr handle, ptr_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    result->err = cudnnGetStream((cudnnHandle_t)handle, (cudaStream_t*)&result->ptr_result_u.ptr);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnncreatetensordescriptor_1_svc(ptr_result *result, struct svc_req *rqstp)
{
    RECORD_VOID_API;
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    result->err = cudnnCreateTensorDescriptor((cudnnTensorDescriptor_t*)&result->ptr_result_u.ptr);
    GSCHED_RELEASE;
    RECORD_RESULT(ptr_result_u, *result);
    return 1;
}

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



int server_cudnn_init(int bypass)
{
    int ret = 0;
    ret &= resource_mg_init(&rm_cudnn, bypass);
    ret &= resource_mg_init(&rm_cudnn_tensors, bypass);
    ret &= resource_mg_init(&rm_cudnn_filters, bypass);
    ret &= resource_mg_init(&rm_cudnn_poolings, bypass);
    ret &= resource_mg_init(&rm_cudnn_activations, bypass);
    ret &= resource_mg_init(&rm_cudnn_lrns, bypass);
    return ret;
}

int server_cudnn_deinit(void)
{
    resource_mg_free(&rm_cudnn);
    resource_mg_free(&rm_cudnn_tensors);
    resource_mg_free(&rm_cudnn_filters);
    resource_mg_free(&rm_cudnn_poolings);
    resource_mg_free(&rm_cudnn_activations);
    resource_mg_free(&rm_cudnn_lrns);
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
    result->err = cudnnQueryRuntimeError(
        (cudnnHandle_t)resource_mg_get(&rm_cudnn, (void*)handle),
        (cudnnStatus_t*)&result->int_result_u.data, (cudnnErrQueryMode_t)mode, tag);
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
    result->err = cudnnCreate((cudnnHandle_t*)&result->ptr_result_u.ptr);
    if (resource_mg_create(&rm_cudnn, (void*)result->ptr_result_u.ptr) != 0) {
        LOGE(LOG_ERROR, "error in resource manager");
    }
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
    *result = cudnnDestroy(
        (cudnnHandle_t)resource_mg_get(&rm_cudnn, (void*)handle));
    // TODO: Remove from resource manager
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
    *result = cudnnSetStream(
        (cudnnHandle_t)resource_mg_get(&rm_cudnn, (void*)handle),
        (cudaStream_t)resource_mg_get(&rm_streams, (void*)streamId));
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnngetstream_1_svc(ptr handle, ptr_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    result->err = cudnnGetStream(
        (cudnnHandle_t)resource_mg_get(&rm_cudnn, (void*)handle),
        (cudaStream_t*)&result->ptr_result_u.ptr);

    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnncreatetensordescriptor_1_svc(ptr_result *result, struct svc_req *rqstp)
{
    RECORD_VOID_API;
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    result->err = cudnnCreateTensorDescriptor((cudnnTensorDescriptor_t*)&result->ptr_result_u.ptr);
    if (resource_mg_create(&rm_cudnn_tensors, (void*)result->ptr_result_u.ptr) != 0) {
        LOGE(LOG_ERROR, "error in resource manager");
    }
    GSCHED_RELEASE;
    RECORD_RESULT(ptr_result_u, *result);
    return 1;
}

bool_t rpc_cudnnsettensor4ddescriptor_1_svc(ptr tensorDesc, int format, int dataType, int n, int c, int h, int w, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnsettensor4ddescriptor_1_argument);
    RECORD_NARG(tensorDesc);
    RECORD_NARG(format);
    RECORD_NARG(dataType);
    RECORD_NARG(n);
    RECORD_NARG(c);
    RECORD_NARG(h);
    RECORD_NARG(w);

    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    *result = cudnnSetTensor4dDescriptor(
        (cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)tensorDesc),
        (cudnnTensorFormat_t)format,
        (cudnnDataType_t)dataType,
        n, c, h, w);
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnnsettensor4ddescriptorex_1_svc(ptr tensorDesc, int dataType, int n, int c, int h, int w, int nStride, int cStride, int hStride, int wStride, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnsettensor4ddescriptorex_1_argument);
    RECORD_NARG(tensorDesc);
    RECORD_NARG(dataType);
    RECORD_NARG(n);
    RECORD_NARG(c);
    RECORD_NARG(h);
    RECORD_NARG(w);
    RECORD_NARG(nStride);
    RECORD_NARG(cStride);
    RECORD_NARG(hStride);
    RECORD_NARG(wStride);

    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    *result = cudnnSetTensor4dDescriptorEx(
        (cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)tensorDesc),
        (cudnnDataType_t)dataType,
        n, c, h, w, nStride, cStride, hStride, wStride);
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnngettensor4ddescriptor_1_svc(ptr tensorDesc, int9_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    result->err = cudnnGetTensor4dDescriptor(
        (cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)tensorDesc),
        (cudnnDataType_t*)&result->int9_result_u.data[0],
        &result->int9_result_u.data[1],
        &result->int9_result_u.data[2],
        &result->int9_result_u.data[3],
        &result->int9_result_u.data[4],
        &result->int9_result_u.data[5],
        &result->int9_result_u.data[6],
        &result->int9_result_u.data[7],
        &result->int9_result_u.data[8]);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnnsettensornddescriptor_1_svc(ptr tensorDesc, int dataType, int nbDims, mem_data dimA, mem_data strideA, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnsettensornddescriptor_1_argument);
    RECORD_NARG(tensorDesc);
    RECORD_NARG(dataType);
    RECORD_NARG(nbDims);
    RECORD_NARG(dimA);
    RECORD_NARG(strideA);
    
    //TODO: Recording dimA and strideA is not as easy as done here.

    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    if (dimA.mem_data_len != nbDims * sizeof(int) || strideA.mem_data_len != nbDims * sizeof(int)) {
        LOGE(LOG_ERROR, "array dimensions not as expected.");
        return 0;
    }
    GSCHED_RETAIN;
    *result = cudnnSetTensorNdDescriptor(
        (cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)tensorDesc),
        (cudnnDataType_t)dataType,
        nbDims,
        (const int*)dimA.mem_data_val,
        (const int*)strideA.mem_data_val);
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnnsettensornddescriptorex_1_svc(ptr tensorDesc, int format, int dataType, int nbDims, mem_data dimA, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnsettensornddescriptorex_1_argument);
    RECORD_NARG(tensorDesc);
    RECORD_NARG(format);
    RECORD_NARG(dataType);
    RECORD_NARG(nbDims);
    RECORD_NARG(dimA);
    
    //TODO: Recording dimA and strideA is not as easy as done here.

    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    if (dimA.mem_data_len != nbDims * sizeof(int)) {
        LOGE(LOG_ERROR, "array dimensions not as expected.");
        return 0;
    }
    GSCHED_RETAIN;
    *result = cudnnSetTensorNdDescriptorEx(
        (cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)tensorDesc),
        (cudnnTensorFormat_t)format,   
        (cudnnDataType_t)dataType,
        nbDims,
        (const int*)dimA.mem_data_val);
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnngettensornddescriptor_1_svc(ptr tensorDesc, int nbDimsRequested, mem_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    result->mem_result_u.data.mem_data_len = sizeof(cudnnDataType_t) + sizeof(int) + nbDimsRequested*sizeof(int)*2;
    if ((result->mem_result_u.data.mem_data_val = malloc(result->mem_result_u.data.mem_data_len)) == NULL) {
        LOGE(LOG_ERROR, "malloc failed");
        return 0;
    }
    
    GSCHED_RETAIN;
    result->err = cudnnGetTensorNdDescriptor(
        (cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)tensorDesc),
        nbDimsRequested,
        (cudnnDataType_t*)result->mem_result_u.data.mem_data_val,
        (int*)&result->mem_result_u.data.mem_data_val[sizeof(cudnnDataType_t)],
        (int*)&result->mem_result_u.data.mem_data_val[sizeof(cudnnDataType_t)+sizeof(int)],
        (int*)&result->mem_result_u.data.mem_data_val[sizeof(cudnnDataType_t)+sizeof(int)+nbDimsRequested*sizeof(int)]);

    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnngettensorsizeinbytes_1_svc(ptr tensorDesc, sz_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    result->err = cudnnGetTensorSizeInBytes(
        (cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)tensorDesc),
        &result->sz_result_u.data);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnndestroytensordescriptor_1_svc(ptr tensorDesc, int *result, struct svc_req *rqstp)
{
    RECORD_API(ptr);
    RECORD_SINGLE_ARG(tensorDesc);
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    *result = cudnnDestroyTensorDescriptor(
        (cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)tensorDesc));
    // TODO: Remove from resource manager
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}


bool_t rpc_cudnncreatefilterdescriptor_1_svc(ptr_result *result, struct svc_req *rqstp)
{
    RECORD_VOID_API;
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    result->err = cudnnCreateFilterDescriptor((cudnnFilterDescriptor_t*)&result->ptr_result_u.ptr);
    if (resource_mg_create(&rm_cudnn_filters, (void*)result->ptr_result_u.ptr) != 0) {
        LOGE(LOG_ERROR, "error in resource manager");
    }
    GSCHED_RELEASE;
    RECORD_RESULT(ptr_result_u, *result);
    return 1;
}

bool_t rpc_cudnnsetfilter4ddescriptor_1_svc(ptr filterDesc, int dataType, int format, int k, int c, int h, int w, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnsetfilter4ddescriptor_1_argument);
    RECORD_NARG(filterDesc);
    RECORD_NARG(dataType);
    RECORD_NARG(format);
    RECORD_NARG(k);
    RECORD_NARG(c);
    RECORD_NARG(h);
    RECORD_NARG(w);

    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    *result = cudnnSetFilter4dDescriptor(
        (cudnnFilterDescriptor_t)resource_mg_get(&rm_cudnn_filters, (void*)filterDesc),
        (cudnnDataType_t)dataType,
        (cudnnTensorFormat_t)format,
        k, c, h, w);
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnngetfilter4ddescriptor_1_svc(ptr filterDesc, int6_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    result->err = cudnnGetFilter4dDescriptor(
        (cudnnFilterDescriptor_t)resource_mg_get(&rm_cudnn_filters, (void*)filterDesc),
        (cudnnDataType_t*)&result->int6_result_u.data[0],
        (cudnnTensorFormat_t*)&result->int6_result_u.data[1],
        &result->int6_result_u.data[2],
        &result->int6_result_u.data[3],
        &result->int6_result_u.data[4],
        &result->int6_result_u.data[5]);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnnsetfilternddescriptor_1_svc(ptr filterDesc, int dataType, int format, int nbDims, mem_data filterDimA, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnsetfilternddescriptor_1_argument);
    RECORD_NARG(filterDesc);
    RECORD_NARG(dataType);
    RECORD_NARG(format);
    RECORD_NARG(nbDims);
    RECORD_NARG(filterDimA);
    
    //TODO: Recording filterDimA is not as easy as done here.

    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    if (filterDimA.mem_data_len != nbDims * sizeof(int)) {
        LOGE(LOG_ERROR, "array dimension not as expected.");
        return 0;
    }
    GSCHED_RETAIN;
    *result = cudnnSetFilterNdDescriptor(
        (cudnnFilterDescriptor_t)resource_mg_get(&rm_cudnn_filters, (void*)filterDesc),
        (cudnnDataType_t)dataType,
        (cudnnTensorFormat_t)format,
        nbDims,
        (const int*)filterDimA.mem_data_val);
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnngetfilternddescriptor_1_svc(ptr filterDesc, int nbDimsRequested, mem_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    result->mem_result_u.data.mem_data_len = sizeof(cudnnDataType_t) + sizeof(cudnnTensorFormat_t) + sizeof(int) + nbDimsRequested*sizeof(int);
    if ((result->mem_result_u.data.mem_data_val = malloc(result->mem_result_u.data.mem_data_len)) == NULL) {
        LOGE(LOG_ERROR, "malloc failed");
        return 0;
    }
    
    GSCHED_RETAIN;
    result->err = cudnnGetFilterNdDescriptor(
        (cudnnFilterDescriptor_t)resource_mg_get(&rm_cudnn_filters, (void*)filterDesc),
        nbDimsRequested,
        (cudnnDataType_t*)result->mem_result_u.data.mem_data_val,
        (cudnnTensorFormat_t*)&result->mem_result_u.data.mem_data_val[sizeof(cudnnDataType_t)],
        (int*)&result->mem_result_u.data.mem_data_val[sizeof(cudnnDataType_t)+sizeof(cudnnTensorDescriptor_t)],
        (int*)&result->mem_result_u.data.mem_data_val[sizeof(cudnnDataType_t)+sizeof(cudnnTensorDescriptor_t)+sizeof(int)]);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnngetfiltersizeinbytes_1_svc(ptr filterDesc, sz_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    result->err = cudnnGetFilterSizeInBytes(
        (cudnnFilterDescriptor_t)resource_mg_get(&rm_cudnn_filters, (void*)filterDesc),
        &result->sz_result_u.data);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnntransformfilter_1_svc(ptr handle, ptr transDesc, cudnn_scaling_t alpha, ptr srcDesc, ptr srcData, cudnn_scaling_t beta, ptr destDesc, ptr destData, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnntransformfilter_1_argument);
    RECORD_NARG(handle);
    RECORD_NARG(transDesc);
    RECORD_NARG(alpha);
    RECORD_NARG(srcDesc);
    RECORD_NARG(srcData);
    RECORD_NARG(beta);
    RECORD_NARG(destDesc);
    RECORD_NARG(destData);
    
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    *result = cudnnTransformFilter(
        (cudnnHandle_t)resource_mg_get(&rm_cudnn, (void*)handle),
        (const cudnnTensorTransformDescriptor_t)resource_mg_get(&rm_cudnn_tensortransform, (void*)transDesc),
        (alpha.dataType == CUDNN_DATA_DOUBLE ? (const void*)&alpha.cudnn_scaling_t_u.d : (const void*)&alpha.cudnn_scaling_t_u.f),
        (const cudnnFilterDescriptor_t)resource_mg_get(&rm_cudnn_filters, (void*)srcDesc),
        (const void*)srcData,
        (beta.dataType == CUDNN_DATA_DOUBLE ? (const void*)&beta.cudnn_scaling_t_u.d : (const void*)&beta.cudnn_scaling_t_u.f),
        (const cudnnFilterDescriptor_t)resource_mg_get(&rm_cudnn_filters, (void*)destDesc),
        (void*)destData);
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnndestroyfilterdescriptor_1_svc(ptr filterDesc, int *result, struct svc_req *rqstp)
{
    RECORD_API(ptr);
    RECORD_SINGLE_ARG(filterDesc);
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    *result = cudnnDestroyFilterDescriptor(
        (cudnnFilterDescriptor_t)resource_mg_get(&rm_cudnn_filters, (void*)filterDesc));
    // TODO: Remove from resource manager
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnncreatepoolingdescriptor_1_svc(ptr_result *result, struct svc_req *rqstp)
{
    RECORD_VOID_API;
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    result->err = cudnnCreatePoolingDescriptor((cudnnPoolingDescriptor_t*)&result->ptr_result_u.ptr);
    if (resource_mg_create(&rm_cudnn_poolings, (void*)result->ptr_result_u.ptr) != 0) {
        LOGE(LOG_ERROR, "error in resource manager");
    }
    GSCHED_RELEASE;
    RECORD_RESULT(ptr_result_u, *result);
    return 1;
}

bool_t rpc_cudnnsetpooling2ddescriptor_1_svc(ptr poolingDesc, int mode, int maxpoolingNanOpt, int windowHeight, int windowWidth, int verticalPadding, int horizontalPadding, int verticalStride, int horizontalStride, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnsetpooling2ddescriptor_1_argument);
    RECORD_NARG(poolingDesc);
    RECORD_NARG(mode);
    RECORD_NARG(maxpoolingNanOpt);
    RECORD_NARG(windowHeight);
    RECORD_NARG(windowWidth);
    RECORD_NARG(verticalPadding);
    RECORD_NARG(horizontalPadding);
    RECORD_NARG(verticalStride);
    RECORD_NARG(horizontalStride);

    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    *result = cudnnSetPooling2dDescriptor(
        (cudnnPoolingDescriptor_t)resource_mg_get(&rm_cudnn_poolings, (void*)poolingDesc),
        (cudnnPoolingMode_t)mode,
        (cudnnNanPropagation_t)maxpoolingNanOpt,
        windowHeight, windowWidth,
        verticalPadding, horizontalPadding,
        verticalStride, horizontalStride);
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnngetpooling2ddescriptor_1_svc(ptr poolingDesc, int8_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    result->err = cudnnGetPooling2dDescriptor(
        (cudnnPoolingDescriptor_t)resource_mg_get(&rm_cudnn_poolings, (void*)poolingDesc),
        (cudnnPoolingMode_t*)&result->int8_result_u.data[0],
        (cudnnNanPropagation_t*)&result->int8_result_u.data[1],
        &result->int8_result_u.data[2],
        &result->int8_result_u.data[3],
        &result->int8_result_u.data[4],
        &result->int8_result_u.data[5],
        &result->int8_result_u.data[6],
        &result->int8_result_u.data[7]);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnnsetpoolingnddescriptor_1_svc(ptr poolingDesc, int mode, int maxpoolingNanOpt, int nbDims, mem_data windowDimA, mem_data paddingA, mem_data strideA, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnsetpoolingnddescriptor_1_argument);
    RECORD_NARG(poolingDesc);
    RECORD_NARG(mode);
    RECORD_NARG(maxpoolingNanOpt);
    RECORD_NARG(nbDims);
    RECORD_NARG(windowDimA);
    RECORD_NARG(paddingA);
    RECORD_NARG(strideA);
    //TODO: Recording windowDimA, paddingA and strideA are not as easy as done here.

    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    if (windowDimA.mem_data_len != nbDims * sizeof(int) ||
        paddingA.mem_data_len != nbDims * sizeof(int) ||
        strideA.mem_data_len != nbDims * sizeof(int)) {
        LOGE(LOG_ERROR, "array dimensions not as expected.");
        return 0;
    }
    GSCHED_RETAIN;
    *result = cudnnSetPoolingNdDescriptor(
        (cudnnPoolingDescriptor_t)resource_mg_get(&rm_cudnn_poolings, (void*)poolingDesc),
        (cudnnPoolingMode_t)mode,
        (cudnnNanPropagation_t)maxpoolingNanOpt,
        nbDims,
        (const int*)windowDimA.mem_data_val,
        (const int*)paddingA.mem_data_val,
        (const int*)strideA.mem_data_val);
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnngetpoolingnddescriptor_1_svc(ptr poolingDesc, int nbDimsRequested, mem_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    result->mem_result_u.data.mem_data_len = sizeof(cudnnPoolingMode_t) + sizeof(cudnnNanPropagation_t) + nbDimsRequested * sizeof(int) * 3;
    if ((result->mem_result_u.data.mem_data_val = malloc(result->mem_result_u.data.mem_data_len)) == NULL) {
        LOGE(LOG_ERROR, "malloc failed");
        return 0;
    }
    
    size_t offsets[] = {
        0,
        sizeof(cudnnPoolingMode_t),
        sizeof(cudnnPoolingMode_t) + sizeof(cudnnNanPropagation_t),
        sizeof(cudnnPoolingMode_t) + sizeof(cudnnNanPropagation_t) + sizeof(int),
        sizeof(cudnnPoolingMode_t) + sizeof(cudnnNanPropagation_t) + sizeof(int) + sizeof(int) * nbDimsRequested,
        sizeof(cudnnPoolingMode_t) + sizeof(cudnnNanPropagation_t) + sizeof(int) + sizeof(int) * nbDimsRequested * 2,
    };
    
    GSCHED_RETAIN;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wint-to-pointer-cast"
    result->err = cudnnGetPoolingNdDescriptor(
        (cudnnPoolingDescriptor_t)resource_mg_get(&rm_cudnn_poolings, (void*)poolingDesc),
        nbDimsRequested,
        (cudnnPoolingMode_t*)result->mem_result_u.data.mem_data_val[offsets[0]],
        (cudnnNanPropagation_t*)result->mem_result_u.data.mem_data_val[offsets[1]],
        (int*)result->mem_result_u.data.mem_data_val[offsets[2]],
        (int*)result->mem_result_u.data.mem_data_val[offsets[3]],
        (int*)result->mem_result_u.data.mem_data_val[offsets[4]],
        (int*)result->mem_result_u.data.mem_data_val[offsets[5]]);
#pragma GCC diagnostic pop

    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnngetpoolingndforwardoutputdim_1_svc(ptr poolingDesc, ptr inputTensorDesc, int nbDims, mem_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    result->mem_result_u.data.mem_data_len = sizeof(int) * nbDims;
    if ((result->mem_result_u.data.mem_data_val = malloc(result->mem_result_u.data.mem_data_len)) == NULL) {
        LOGE(LOG_ERROR, "malloc failed");
        return 0;
    }
    result->err = cudnnGetPoolingNdForwardOutputDim(
        (cudnnPoolingDescriptor_t)resource_mg_get(&rm_cudnn_poolings, (void*)poolingDesc),
        (cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)inputTensorDesc),
        nbDims,
        (int*)&result->mem_result_u.data);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnngetpooling2dforwardoutputdim_1_svc(ptr poolingDesc, ptr inputTensorDesc, int4_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    result->err = cudnnGetPooling2dForwardOutputDim(
        (cudnnPoolingDescriptor_t)resource_mg_get(&rm_cudnn_poolings, (void*)poolingDesc),
        (cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)inputTensorDesc),
        (int*)&result->int4_result_u.data[0],
        (int*)&result->int4_result_u.data[1],
        (int*)&result->int4_result_u.data[2],
        (int*)&result->int4_result_u.data[3]);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnndestroypoolingdescriptor_1_svc(ptr poolingDesc, int *result, struct svc_req *rqstp)
{
    RECORD_API(ptr);
    RECORD_SINGLE_ARG(poolingDesc);
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    *result = cudnnDestroyPoolingDescriptor(
        (cudnnPoolingDescriptor_t)resource_mg_get(&rm_cudnn_poolings, (void*)poolingDesc));
    // TODO: Remove from resource manager
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnncreateactivationdescriptor_1_svc(ptr_result *result, struct svc_req *rqstp)
{
    RECORD_VOID_API;
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    result->err = cudnnCreateActivationDescriptor((cudnnActivationDescriptor_t*)&result->ptr_result_u.ptr);
    if (resource_mg_create(&rm_cudnn_activations, (void*)result->ptr_result_u.ptr) != 0) {
        LOGE(LOG_ERROR, "error in resource manager");
    }
    GSCHED_RELEASE;
    RECORD_RESULT(ptr_result_u, *result);
    return 1;
}

bool_t rpc_cudnnsetactivationdescriptor_1_svc(ptr activationDesc, int mode, int reluNanOpt, double coef, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnsetactivationdescriptor_1_argument);
    RECORD_NARG(activationDesc);
    RECORD_NARG(mode);
    RECORD_NARG(reluNanOpt);
    RECORD_NARG(coef);

    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    *result = cudnnSetActivationDescriptor(
        (cudnnActivationDescriptor_t)resource_mg_get(&rm_cudnn_activations, (void*)activationDesc),
        (cudnnActivationMode_t)mode,
        (cudnnNanPropagation_t)reluNanOpt,
        coef);
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnngetactivationdescriptor_1_svc(ptr activationDesc, int2d1_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    result->err = cudnnGetActivationDescriptor(
        (cudnnActivationDescriptor_t)resource_mg_get(&rm_cudnn_activations, (void*)activationDesc),
        (cudnnActivationMode_t*)&result->int2d1_result_u.data.i[0],
        (cudnnNanPropagation_t*)&result->int2d1_result_u.data.i[1],
        &result->int2d1_result_u.data.d);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnnsetactivationdescriptorswishbeta_1_svc(ptr activationDesc, double swish_beta, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnsetactivationdescriptorswishbeta_1_argument);
    RECORD_NARG(activationDesc);
    RECORD_NARG(swish_beta);

    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    *result = cudnnSetActivationDescriptorSwishBeta(
        (cudnnActivationDescriptor_t)resource_mg_get(&rm_cudnn_activations, (void*)activationDesc),
        swish_beta);
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnngetactivationdescriptorswishbeta_1_svc(ptr activationDesc, d_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    result->err = cudnnGetActivationDescriptorSwishBeta(
        (cudnnActivationDescriptor_t)resource_mg_get(&rm_cudnn_activations, (void*)activationDesc),
        &result->d_result_u.data);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnndestroyactivationdescriptor_1_svc(ptr activationDesc, int *result, struct svc_req *rqstp)
{
    RECORD_API(ptr);
    RECORD_SINGLE_ARG(activationDesc);
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    *result = cudnnDestroyActivationDescriptor(
        (cudnnActivationDescriptor_t)resource_mg_get(&rm_cudnn_activations, (void*)activationDesc));
    // TODO: Remove from resource manager
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnncreatelrndescriptor_1_svc(ptr_result *result, struct svc_req *rqstp)
{
    RECORD_VOID_API;
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    result->err = cudnnCreateLRNDescriptor((cudnnLRNDescriptor_t*)&result->ptr_result_u.ptr);
    if (resource_mg_create(&rm_cudnn_lrns, (void*)result->ptr_result_u.ptr) != 0) {
        LOGE(LOG_ERROR, "error in resource manager");
    }
    GSCHED_RELEASE;
    RECORD_RESULT(ptr_result_u, *result);
    return 1;
}

bool_t rpc_cudnnsetlrndescriptor_1_svc(ptr normDesc, unsigned lrnN, double lrnAlpha, double lrnBeta, double lrnK, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnsetlrndescriptor_1_argument);
    RECORD_NARG(normDesc);
    RECORD_NARG(lrnN);
    RECORD_NARG(lrnAlpha);
    RECORD_NARG(lrnBeta);
    RECORD_NARG(lrnK);

    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    *result = cudnnSetLRNDescriptor(
        (cudnnLRNDescriptor_t)resource_mg_get(&rm_cudnn_lrns, (void*)normDesc),
        lrnN,
        lrnAlpha,
        lrnBeta,
        lrnK);
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnngetlrndescriptor_1_svc(ptr normDesc, int1d3_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    result->err = cudnnGetLRNDescriptor(
        (cudnnLRNDescriptor_t)resource_mg_get(&rm_cudnn_lrns, (void*)normDesc),
        (unsigned int*)&result->int1d3_result_u.data.i,
        &result->int1d3_result_u.data.d[0],
        &result->int1d3_result_u.data.d[1],
        &result->int1d3_result_u.data.d[2]);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnndestroylrndescriptor_1_svc(ptr lrnDesc, int *result, struct svc_req *rqstp)
{
    RECORD_API(ptr);
    RECORD_SINGLE_ARG(lrnDesc);
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    *result = cudnnDestroyLRNDescriptor(
        (cudnnLRNDescriptor_t)resource_mg_get(&rm_cudnn_lrns, (void*)lrnDesc));
    // TODO: Remove from resource manager
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}
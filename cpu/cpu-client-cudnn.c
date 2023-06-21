#include <cuda_runtime.h>
#include <cudnn.h>
#include <stdint.h>

#include "cpu-libwrap.h"
#include "cpu_rpc_prot.h"
#include "cpu-common.h"
#include "cpu-utils.h"
#include "log.h"

#ifdef WITH_API_CNT
extern int api_call_cnt;
#endif //WITH_API_CNT

size_t cudnnGetVersion(void)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    size_t result;
    enum clnt_stat retval_1;
    retval_1 = rpc_cudnngetversion_1(&result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (%d)", __FUNCTION__, retval_1);
    }
    return result;
}
size_t cudnnGetMaxDeviceVersion(void)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    size_t result;
    enum clnt_stat retval_1;
    retval_1 = rpc_cudnngetmaxdeviceversion_1(&result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (%d)", __FUNCTION__, retval_1);
    }
    return result;
}
size_t cudnnGetCudartVersion(void)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    size_t result;
    enum clnt_stat retval_1;
    retval_1 = rpc_cudnngetcudartversion_1(&result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (%d)", __FUNCTION__, retval_1);
    }
    return result;
}
const char *cudnnGetErrorString(cudnnStatus_t status)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    char *result;
    enum clnt_stat retval_1;
    retval_1 = rpc_cudnngeterrorstring_1((int)status, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (%d)", __FUNCTION__, retval_1);
    }
    if (result == NULL) {
        LOGE(LOG_ERROR, "%s failed (result is NULL)", __FUNCTION__);
    }
    return result;
}

cudnnStatus_t cudnnQueryRuntimeError(cudnnHandle_t handle, cudnnStatus_t* rstatus, cudnnErrQueryMode_t  mode, cudnnRuntimeTag_t * tag)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int_result result;
    enum clnt_stat retval_1;
    retval_1 = rpc_cudnnqueryruntimeerror_1((ptr)handle, (int)mode, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (%d)", __FUNCTION__, retval_1);
    }
    if (result.err != CUDNN_STATUS_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (result is %d)", __FUNCTION__, result.err);
    } else {
        *rstatus = (cudnnStatus_t)result.int_result_u.data;
        //*tag = NULL;
    }
    return result.err;
}

cudnnStatus_t cudnnGetProperty(libraryPropertyType type, int * value)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int_result result;
    enum clnt_stat retval_1;
    if (value == NULL) {
        LOGE(LOG_ERROR, "%s failed (value is NULL)", __FUNCTION__);
        return CUDNN_STATUS_BAD_PARAM;
    }
    retval_1 = rpc_cudnngetproperty_1((int)type, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (%d)", __FUNCTION__, retval_1);
    }
    if (result.err != CUDNN_STATUS_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (result is %d)", __FUNCTION__, result.err);
    } else {
        *value = result.int_result_u.data;
    }
    return result.err;
}

cudnnStatus_t cudnnCreate(cudnnHandle_t* handle)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    ptr_result result;
    enum clnt_stat retval_1;
    if (handle == NULL) {
        LOGE(LOG_ERROR, "%s failed (value is NULL)", __FUNCTION__);
        return CUDNN_STATUS_BAD_PARAM;
    }
    retval_1 = rpc_cudnncreate_1(&result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (%d)", __FUNCTION__, retval_1);
    }
    if (result.err != CUDNN_STATUS_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (result is %d)", __FUNCTION__, result.err);
    } else {
        *handle = (cudnnHandle_t)result.ptr_result_u.ptr;
    }
    return result.err;
}

cudnnStatus_t cudnnDestroy(cudnnHandle_t handle)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval_1;
    retval_1 = rpc_cudnndestroy_1((ptr)handle, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (%d)", __FUNCTION__, retval_1);
    }
    if (result != CUDNN_STATUS_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (result is %d)", __FUNCTION__, result);
    }
    return result;
}

cudnnStatus_t cudnnSetStream(cudnnHandle_t handle, cudaStream_t streamId)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval_1;
    retval_1 = rpc_cudnnsetstream_1((ptr)handle, (ptr)streamId, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (%d)", __FUNCTION__, retval_1);
    }
    if (result != CUDNN_STATUS_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (result is %d)", __FUNCTION__, result);
    }
    return result;
}

cudnnStatus_t cudnnGetStream(cudnnHandle_t handle, cudaStream_t * streamId)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    ptr_result result;
    enum clnt_stat retval_1;
    if (streamId == NULL) {
        LOGE(LOG_ERROR, "%s failed (value is NULL)", __FUNCTION__);
        return CUDNN_STATUS_BAD_PARAM;
    }
    retval_1 = rpc_cudnngetstream_1((ptr)handle, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (%d)", __FUNCTION__, retval_1);
    }
    if (result.err != CUDNN_STATUS_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (result is %d)", __FUNCTION__, result.err);
    } else {
        *streamId = (cudaStream_t)result.ptr_result_u.ptr;
    }
    return result.err;
}

cudnnStatus_t cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t * tensorDesc)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    ptr_result result;
    enum clnt_stat retval_1;
    if (tensorDesc == NULL) {
        LOGE(LOG_ERROR, "%s failed (value is NULL)", __FUNCTION__);
        return CUDNN_STATUS_BAD_PARAM;
    }
    retval_1 = rpc_cudnncreatetensordescriptor_1(&result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (%d)", __FUNCTION__, retval_1);
    }
    if (result.err != CUDNN_STATUS_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (result is %d)", __FUNCTION__, result.err);
    } else {
        *tensorDesc = (cudnnTensorDescriptor_t)result.ptr_result_u.ptr;
    }
    return result.err;
}

DEF_FN(cudnnStatus_t, cudnnSetTensor4dDescriptor, cudnnTensorDescriptor_t, tensorDesc, cudnnTensorFormat_t, format, cudnnDataType_t, dataType, int, n, int, c, int, h, int, w) 
DEF_FN(cudnnStatus_t, cudnnSetTensor4dDescriptorEx, cudnnTensorDescriptor_t, tensorDesc, cudnnDataType_t, dataType, int, n, int, c, int, h, int, w, int, nStride, int, cStride, int, hStride, int, wStride)
DEF_FN(cudnnStatus_t, cudnnGetTensor4dDescriptor, const cudnnTensorDescriptor_t, tensorDesc, cudnnDataType_t *, dataType, int*, n, int*, c, int*, h, int*, w, int*, nStride, int*, cStride, int*, hStride, int*, wStride)
DEF_FN(cudnnStatus_t, cudnnSetTensorNdDescriptor, cudnnTensorDescriptor_t, tensorDesc, cudnnDataType_t, dataType, int, nbDims, const int*, dimA, const int*, strideA)
DEF_FN(cudnnStatus_t, cudnnSetTensorNdDescriptorEx, cudnnTensorDescriptor_t, tensorDesc, cudnnTensorFormat_t, format, cudnnDataType_t, dataType, int, nbDims, const int*, dimA)
DEF_FN(cudnnStatus_t, cudnnGetTensorNdDescriptor, const cudnnTensorDescriptor_t, tensorDesc, int, nbDimsRequested, cudnnDataType_t *, dataType, int*, nbDims, int*, dimA, int*, strideA)
DEF_FN(cudnnStatus_t, cudnnGetTensorSizeInBytes, const cudnnTensorDescriptor_t, tensorDesc, size_t*, size)
DEF_FN(cudnnStatus_t, cudnnDestroyTensorDescriptor, cudnnTensorDescriptor_t, tensorDesc)
DEF_FN(cudnnStatus_t, cudnnInitTransformDest, const cudnnTensorTransformDescriptor_t, transformDesc, const cudnnTensorDescriptor_t, srcDesc, cudnnTensorDescriptor_t, destDesc, size_t*, destSizeInBytes)
DEF_FN(cudnnStatus_t, cudnnCreateTensorTransformDescriptor, cudnnTensorTransformDescriptor_t *, transformDesc)
DEF_FN(cudnnStatus_t, cudnnSetTensorTransformDescriptor, cudnnTensorTransformDescriptor_t, transformDesc, const uint32_t, nbDims, const cudnnTensorFormat_t, destFormat, const int32_t*, padBeforeA, const int32_t*, padAfterA, const uint32_t*, foldA, const cudnnFoldingDirection_t,  direction)
DEF_FN(cudnnStatus_t, cudnnGetTensorTransformDescriptor, cudnnTensorTransformDescriptor_t, transformDesc, uint32_t, nbDimsRequested, cudnnTensorFormat_t *, destFormat, int32_t*, padBeforeA, int32_t*, padAfterA, uint32_t*, foldA, cudnnFoldingDirection_t *, direction)
DEF_FN(cudnnStatus_t, cudnnDestroyTensorTransformDescriptor, cudnnTensorTransformDescriptor_t, transformDesc)
DEF_FN(cudnnStatus_t, cudnnTransformTensor, cudnnHandle_t, handle, const void *, alpha, const cudnnTensorDescriptor_t, xDesc, const void *, x, const void *, beta, const cudnnTensorDescriptor_t, yDesc, void *, y)
DEF_FN(cudnnStatus_t, cudnnTransformTensorEx, cudnnHandle_t, handle, const cudnnTensorTransformDescriptor_t, transDesc, const void *, alpha, const cudnnTensorDescriptor_t, srcDesc, const void *, srcData, const void *, beta, const cudnnTensorDescriptor_t, destDesc, void *, destData)
DEF_FN(cudnnStatus_t, cudnnAddTensor, cudnnHandle_t, handle, const void *, alpha, const cudnnTensorDescriptor_t, aDesc, const void *, A, const void *, ,beta, const cudnnTensorDescriptor_t, cDesc, void *, C)
DEF_FN(cudnnStatus_t, cudnnCreateOpTensorDescriptor, cudnnOpTensorDescriptor_t *, opTensorDesc)
DEF_FN(cudnnStatus_t, cudnnSetOpTensorDescriptor, cudnnOpTensorDescriptor_t, opTensorDesc, cudnnOpTensorOp_t, opTensorOp, cudnnDataType_t, opTensorCompType, cudnnNanPropagation_t, opTensorNanOpt)
DEF_FN(cudnnStatus_t, cudnnGetOpTensorDescriptor, const cudnnOpTensorDescriptor_t, opTensorDesc, cudnnOpTensorOp_t *, opTensorOp, cudnnDataType_t *, opTensorCompType, cudnnNanPropagation_t *, opTensorNanOpt)
DEF_FN(cudnnStatus_t, cudnnDestroyOpTensorDescriptor, cudnnOpTensorDescriptor_t, opTensorDesc)
DEF_FN(cudnnStatus_t, cudnnOpTensor, cudnnHandle_t, handle, const cudnnOpTensorDescriptor_t, opTensorDesc, const void *, alpha1, const cudnnTensorDescriptor_t, aDesc, const void *, A, const void *, alpha2, const cudnnTensorDescriptor_t, bDesc, const void *, B, const void *, beta, const cudnnTensorDescriptor_t,  cDesc, void *, C)
DEF_FN(cudnnStatus_t, cudnnCreateReduceTensorDescriptor, cudnnReduceTensorDescriptor_t *, reduceTensorDesc)
DEF_FN(cudnnStatus_t, cudnnSetReduceTensorDescriptor, cudnnReduceTensorDescriptor_t, reduceTensorDesc, cudnnReduceTensorOp_t, reduceTensorOp, cudnnDataType_t, reduceTensorCompType, cudnnNanPropagation_t, reduceTensorNanOpt, cudnnReduceTensorIndices_t, reduceTensorIndices, cudnnIndicesType_t, reduceTensorIndicesType)
DEF_FN(cudnnStatus_t, cudnnGetReduceTensorDescriptor, const cudnnReduceTensorDescriptor_t, reduceTensorDesc, cudnnReduceTensorOp_t *, reduceTensorOp, cudnnDataType_t *, reduceTensorCompType, cudnnNanPropagation_t *, reduceTensorNanOpt, cudnnReduceTensorIndices_t *, reduceTensorIndices, cudnnIndicesType_t *, reduceTensorIndicesType)
DEF_FN(cudnnStatus_t, cudnnDestroyReduceTensorDescriptor, cudnnReduceTensorDescriptor_t, reduceTensorDesc)
DEF_FN(cudnnStatus_t, cudnnGetReductionIndicesSize, cudnnHandle_t, handle, const cudnnReduceTensorDescriptor_t, reduceTensorDesc, const cudnnTensorDescriptor_t, aDesc, const cudnnTensorDescriptor_t, cDesc, size_t*, sizeInBytes)
DEF_FN(cudnnStatus_t, cudnnGetReductionWorkspaceSize, cudnnHandle_t, handle, const cudnnReduceTensorDescriptor_t, reduceTensorDesc, const cudnnTensorDescriptor_t, aDesc, const cudnnTensorDescriptor_t, cDesc, size_t*, sizeInBytes)
DEF_FN(cudnnStatus_t, cudnnReduceTensor, cudnnHandle_t, handle, const cudnnReduceTensorDescriptor_t, reduceTensorDesc, void *, indices, size_t, indicesSizeInBytes, void *, workspace, size_t, workspaceSizeInBytes, const void *, alpha, const cudnnTensorDescriptor_t, aDesc, const void *, A, const void *, beta, const cudnnTensorDescriptor_t, cDesc, void *, C)
DEF_FN(cudnnStatus_t, cudnnSetTensor, cudnnHandle_t, handle, const cudnnTensorDescriptor_t, yDesc, void *, y, const void *, valuePtr)
DEF_FN(cudnnStatus_t, cudnnScaleTensor, cudnnHandle_t, handle, const cudnnTensorDescriptor_t, yDesc, void *, y, const void *, alpha)

cudnnStatus_t cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t * filterDesc)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    ptr_result result;
    enum clnt_stat retval_1;
    if (filterDesc == NULL) {
        LOGE(LOG_ERROR, "%s failed (value is NULL)", __FUNCTION__);
        return CUDNN_STATUS_BAD_PARAM;
    }
    retval_1 = rpc_cudnncreatefilterdescriptor_1(&result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (%d)", __FUNCTION__, retval_1);
    }
    if (result.err != CUDNN_STATUS_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (result is %d)", __FUNCTION__, result.err);
    } else {
        *filterDesc = (cudnnFilterDescriptor_t)result.ptr_result_u.ptr;
    }
    return result.err;
}

cudnnStatus_t cudnnSetFilter4dDescriptor(cudnnFilterDescriptor_t filterDesc, cudnnDataType_t dataType, cudnnTensorFormat_t format, int k, int c, int h, int w) 
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval_1;
    retval_1 = rpc_cudnnsetfilter4ddescriptor_1(
        (ptr)filterDesc,
        (int)dataType,
        (int)format,
        k, c, h, w, &result, clnt);

    if (retval_1 != RPC_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (%d)", __FUNCTION__, retval_1);
    }
    if (result != CUDNN_STATUS_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (result is %d)", __FUNCTION__, result);
    } 
    return result;
}

cudnnStatus_t cudnnGetFilter4dDescriptor(const cudnnFilterDescriptor_t filterDesc, cudnnDataType_t *dataType, cudnnTensorFormat_t *format, int* k, int* c, int* h, int* w) 
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int6_result result;
    enum clnt_stat retval_1;
    if (dataType == NULL || format == NULL || k == NULL || c == NULL || h == NULL || w == NULL) {
        LOGE(LOG_ERROR, "%s failed (value is NULL)", __FUNCTION__);
        return CUDNN_STATUS_BAD_PARAM;
    }
    retval_1 = rpc_cudnngetfilter4ddescriptor_1(
        (ptr)filterDesc,
        &result, clnt);

    if (retval_1 != RPC_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (%d)", __FUNCTION__, retval_1);
    }
    if (result.err != CUDNN_STATUS_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (result is %d)", __FUNCTION__, result);
    } else {
        *dataType = (cudnnDataType_t)result.int6_result_u.data[0];
        *format = (cudnnTensorFormat_t)result.int6_result_u.data[1];
        *k = result.int6_result_u.data[2];
        *c = result.int6_result_u.data[3];
        *h = result.int6_result_u.data[4];
        *w = result.int6_result_u.data[5];
    }
    return result.err;
}

cudnnStatus_t cudnnSetFilterNdDescriptor(cudnnFilterDescriptor_t filterDesc, cudnnDataType_t dataType, cudnnTensorFormat_t format, int nbDims, const int* filterDimA)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval_1;
    mem_data rpc_filterDimA = {
        .mem_data_len = nbDims * sizeof(int),
        .mem_data_val = (char*)filterDimA
    };
    retval_1 = rpc_cudnnsetfilternddescriptor_1(
        (ptr)filterDesc,
        (int)dataType,
        (int)format,
        (int)nbDims,
        rpc_filterDimA, &result, clnt);

    if (retval_1 != RPC_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (%d)", __FUNCTION__, retval_1);
    }
    if (result != CUDNN_STATUS_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (result is %d)", __FUNCTION__, result);
    } 
    return result;
}

cudnnStatus_t cudnnGetFilterNdDescriptor(const cudnnFilterDescriptor_t filterDesc, int nbDimsRequested, cudnnDataType_t * dataType, cudnnTensorFormat_t * format, int* nbDims, int* filterDimA)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    mem_result result;
    enum clnt_stat retval_1;
    if (dataType == NULL || format == NULL || nbDims == NULL || filterDimA == NULL) {
        LOGE(LOG_ERROR, "%s failed (value is NULL)", __FUNCTION__);
        return CUDNN_STATUS_BAD_PARAM;
    }
    retval_1 = rpc_cudnngetfilternddescriptor_1(
        (ptr)filterDesc,
        nbDimsRequested,
        &result, clnt);

    if (retval_1 != RPC_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (%d)", __FUNCTION__, retval_1);
    }
    size_t expected_size = nbDimsRequested * sizeof(int) + sizeof(int) + sizeof(cudnnDataType_t) + sizeof(cudnnTensorFormat_t);
    if (result.err != CUDNN_STATUS_SUCCESS || result.mem_result_u.data.mem_data_len < expected_size) {
        LOGE(LOG_ERROR, "%s failed (result is %d)", __FUNCTION__, result.err);
    } else {
        size_t offset = 0;
        *dataType = (cudnnDataType_t)result.mem_result_u.data.mem_data_val[offset];
        offset += sizeof(cudnnDataType_t);
        *format = (cudnnTensorFormat_t)result.mem_result_u.data.mem_data_val[offset];
        offset += sizeof(cudnnTensorFormat_t);
        *nbDims = (int)result.mem_result_u.data.mem_data_val[offset];
        offset += sizeof(int);
        memcpy(filterDimA, result.mem_result_u.data.mem_data_val+offset, *nbDims * sizeof(int));
    }
    return result.err;
}

cudnnStatus_t cudnnGetFilterSizeInBytes(const cudnnFilterDescriptor_t filterDesc, size_t* size)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    sz_result result;
    enum clnt_stat retval_1;
    if (size == NULL) {
        LOGE(LOG_ERROR, "%s failed (value is NULL)", __FUNCTION__);
        return CUDNN_STATUS_BAD_PARAM;
    }
    retval_1 = rpc_cudnngetfiltersizeinbytes_1(
        (ptr)filterDesc,
        &result, clnt);

    if (retval_1 != RPC_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (%d)", __FUNCTION__, retval_1);
    }
    if (result.err != CUDNN_STATUS_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (result is %d)", __FUNCTION__, result.err);
    } else {
        *size = result.sz_result_u.data;
    }
    return result.err;
}

cudnnStatus_t cudnnTransformFilter(cudnnHandle_t handle, const cudnnTensorTransformDescriptor_t transDesc, const void * alpha, const cudnnFilterDescriptor_t srcDesc, const void * srcData, const void * beta, const cudnnFilterDescriptor_t destDesc, void * destData)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval_1;
    //TODO: Check if we have a float instead of always sending doubles
    cudnn_scaling_t rpc_alpha = {.dataType = CUDNN_DATA_DOUBLE, .cudnn_scaling_t_u.d = *((double*)alpha)};
    cudnn_scaling_t rpc_beta = {.dataType = CUDNN_DATA_DOUBLE, .cudnn_scaling_t_u.d = *((double*)beta)};
    retval_1 = rpc_cudnntransformfilter_1(
        (ptr)handle,
        (ptr)transDesc,
        rpc_alpha,
        (ptr)srcDesc,
        (ptr)srcData,
        rpc_beta,
        (ptr)destDesc,
        (ptr)destData,
        &result, clnt);

    if (retval_1 != RPC_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (%d)", __FUNCTION__, retval_1);
    }
    if (result != CUDNN_STATUS_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (result is %d)", __FUNCTION__, result);
    }
    return result;
}

cudnnStatus_t cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t filterDesc)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval_1;
    retval_1 = rpc_cudnndestroyfilterdescriptor_1(
        (ptr)filterDesc,
        &result, clnt);

    if (retval_1 != RPC_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (%d)", __FUNCTION__, retval_1);
    }
    if (result != CUDNN_STATUS_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (result is %d)", __FUNCTION__, result);
    }
    return result;
}

DEF_FN(cudnnStatus_t, cudnnSoftmaxForward, cudnnHandle_t, handle, cudnnSoftmaxAlgorithm_t, algo, cudnnSoftmaxMode_t, mode, const void *,alpha, const cudnnTensorDescriptor_t, xDesc, const void *, x, const void *, beta, const cudnnTensorDescriptor_t, yDesc, void *, y)
cudnnStatus_t cudnnCreatePoolingDescriptor(cudnnPoolingDescriptor_t *poolingDesc)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    ptr_result result;
    enum clnt_stat retval_1;
    if (poolingDesc == NULL) {
        LOGE(LOG_ERROR, "%s failed (value is NULL)", __FUNCTION__);
        return CUDNN_STATUS_BAD_PARAM;
    }
    retval_1 = rpc_cudnncreatepoolingdescriptor_1(&result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (%d)", __FUNCTION__, retval_1);
    }
    if (result.err != CUDNN_STATUS_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (result is %d)", __FUNCTION__, result.err);
    } else {
        *poolingDesc = (cudnnPoolingDescriptor_t)result.ptr_result_u.ptr;
    }
    return result.err;
}

cudnnStatus_t cudnnSetPooling2dDescriptor(cudnnPoolingDescriptor_t poolingDesc, cudnnPoolingMode_t mode, cudnnNanPropagation_t maxpoolingNanOpt, int windowHeight, int windowWidth, int verticalPadding, int horizontalPadding, int verticalStride, int horizontalStride)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval_1;
    retval_1 = rpc_cudnnsetpooling2ddescriptor_1(
        (ptr)poolingDesc,
        (int)mode,
        (int)maxpoolingNanOpt,
        windowHeight,
        windowWidth,
        verticalPadding,
        horizontalPadding,
        verticalStride,
        horizontalStride,
        &result, clnt);

    if (retval_1 != RPC_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (%d)", __FUNCTION__, retval_1);
    }
    if (result != CUDNN_STATUS_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (result is %d)", __FUNCTION__, result);
    } 
    return result;
}
    
cudnnStatus_t cudnnGetPooling2dDescriptor(const cudnnPoolingDescriptor_t poolingDesc, cudnnPoolingMode_t *mode, cudnnNanPropagation_t *maxpoolingNanOpt, int* windowHeight, int* windowWidth, int* verticalPadding, int* horizontalPadding, int* verticalStride, int* horizontalStride)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int8_result result;
    enum clnt_stat retval_1;
    if (mode == NULL || maxpoolingNanOpt == NULL || windowHeight == NULL || windowWidth == NULL || verticalPadding == NULL || verticalStride == NULL || horizontalPadding == NULL || horizontalStride == NULL) {
        LOGE(LOG_ERROR, "%s failed (value is NULL)", __FUNCTION__);
        return CUDNN_STATUS_BAD_PARAM;
    }
    retval_1 = rpc_cudnngetpooling2ddescriptor_1(
        (ptr)poolingDesc,
        &result, clnt);

    if (retval_1 != RPC_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (%d)", __FUNCTION__, retval_1);
    }
    if (result.err != CUDNN_STATUS_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (result is %d)", __FUNCTION__, result);
    } else {
        *mode = (cudnnPoolingMode_t)result.int8_result_u.data[0];
        *maxpoolingNanOpt = (cudnnNanPropagation_t)result.int8_result_u.data[1];
        *windowHeight = result.int8_result_u.data[2];
        *windowWidth = result.int8_result_u.data[3];
        *verticalPadding = result.int8_result_u.data[4];
        *horizontalPadding = result.int8_result_u.data[5];
        *verticalStride = result.int8_result_u.data[6];
        *horizontalStride = result.int8_result_u.data[7];
    }
    return result.err;
}

cudnnStatus_t cudnnSetPoolingNdDescriptor(cudnnPoolingDescriptor_t poolingDesc, const cudnnPoolingMode_t mode, const cudnnNanPropagation_t maxpoolingNanOpt, int nbDims, const int* windowDimA, const int* paddingA, const int* strideA)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval_1;
    mem_data rpc_windowDimA = {
        .mem_data_len = nbDims * sizeof(int),
        .mem_data_val = (char*)windowDimA
    };
    mem_data rpc_paddingA = {
        .mem_data_len = nbDims * sizeof(int),
        .mem_data_val = (char*)paddingA
    };
    mem_data rpc_strideA = {
        .mem_data_len = nbDims * sizeof(int),
        .mem_data_val = (char*)strideA
    };
    retval_1 = rpc_cudnnsetpoolingnddescriptor_1(
        (ptr)poolingDesc,
        (int)mode,
        (int)maxpoolingNanOpt,
        (int)nbDims,
        rpc_windowDimA,
        rpc_paddingA,
        rpc_strideA,
        &result, clnt);

    if (retval_1 != RPC_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (%d)", __FUNCTION__, retval_1);
    }
    if (result != CUDNN_STATUS_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (result is %d)", __FUNCTION__, result);
    } 
    return result;
}

cudnnStatus_t cudnnGetPoolingNdDescriptor(const cudnnPoolingDescriptor_t poolingDesc, int nbDimsRequested, cudnnPoolingMode_t * mode, cudnnNanPropagation_t * maxpoolingNanOpt, int* nbDims, int* windowDimA, int* paddingA, int* strideA)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    mem_result result;
    enum clnt_stat retval_1;
    if (mode == NULL || maxpoolingNanOpt == NULL || nbDims == NULL || windowDimA == NULL || paddingA == NULL || strideA == NULL) {
        LOGE(LOG_ERROR, "%s failed (value is NULL)", __FUNCTION__);
        return CUDNN_STATUS_BAD_PARAM;
    }
    retval_1 = rpc_cudnngetpoolingnddescriptor_1(
        (ptr)poolingDesc,
        nbDimsRequested,
        &result, clnt); 
    if (retval_1 != RPC_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (%d)", __FUNCTION__, retval_1);
    }
    size_t expected_size = nbDimsRequested * sizeof(int) * 3 + sizeof(int) + sizeof(cudnnPoolingMode_t) + sizeof(cudnnNanPropagation_t);
    if (result.err != CUDNN_STATUS_SUCCESS || result.mem_result_u.data.mem_data_len != expected_size) {
        LOGE(LOG_ERROR, "%s failed (result is %d)", __FUNCTION__, result.err);
    } else {
        size_t offset = 0;
        *mode = (cudnnPoolingMode_t)result.mem_result_u.data.mem_data_val[offset];
        offset += sizeof(cudnnPoolingMode_t);
        *maxpoolingNanOpt = (cudnnNanPropagation_t)result.mem_result_u.data.mem_data_val[offset];
        offset += sizeof(cudnnNanPropagation_t);
        *nbDims = (int)result.mem_result_u.data.mem_data_val[offset];
        offset += sizeof(int);
        memcpy(windowDimA, result.mem_result_u.data.mem_data_val+offset, *nbDims * sizeof(int));
        offset += *nbDims * sizeof(int);
        memcpy(paddingA, result.mem_result_u.data.mem_data_val+offset, *nbDims * sizeof(int));
        offset += *nbDims * sizeof(int);
        memcpy(strideA, result.mem_result_u.data.mem_data_val+offset, *nbDims * sizeof(int));
    }
    return result.err;
}

cudnnStatus_t cudnnGetPoolingNdForwardOutputDim(const cudnnPoolingDescriptor_t poolingDesc, const cudnnTensorDescriptor_t inputTensorDesc, int nbDims, int* outputTensorDimA)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    mem_result result;
    enum clnt_stat retval_1;
    if (outputTensorDimA == NULL) {
        LOGE(LOG_ERROR, "%s failed (value is NULL)", __FUNCTION__);
        return CUDNN_STATUS_BAD_PARAM;
    }
    retval_1 = rpc_cudnngetpoolingndforwardoutputdim_1(
        (ptr)poolingDesc,
        (ptr)inputTensorDesc,
        nbDims,
        &result, clnt); 
    if (retval_1 != RPC_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (%d)", __FUNCTION__, retval_1);
    }
    size_t expected_size = nbDims * sizeof(int);
    if (result.err != CUDNN_STATUS_SUCCESS || result.mem_result_u.data.mem_data_len != expected_size) {
        LOGE(LOG_ERROR, "%s failed (result is %d)", __FUNCTION__, result.err);
    } else {
        memcpy(outputTensorDimA, result.mem_result_u.data.mem_data_val, nbDims * sizeof(int));
    }
    return result.err;
}

cudnnStatus_t cudnnGetPooling2dForwardOutputDim(const cudnnPoolingDescriptor_t poolingDesc, const cudnnTensorDescriptor_t inputTensorDesc, int* n, int* c, int* h, int* w)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int4_result result;
    enum clnt_stat retval_1;
    if (n == NULL || c == NULL || h == NULL || w == NULL) {
        LOGE(LOG_ERROR, "%s failed (value is NULL)", __FUNCTION__);
        return CUDNN_STATUS_BAD_PARAM;
    }
    retval_1 = rpc_cudnngetpooling2dforwardoutputdim_1(
        (ptr)poolingDesc,
        (ptr)inputTensorDesc,
        &result, clnt); 
    if (retval_1 != RPC_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (%d)", __FUNCTION__, retval_1);
    }
    if (result.err != CUDNN_STATUS_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (result is %d)", __FUNCTION__, result.err);
    } else {
        *n = result.int4_result_u.data[0];
        *c = result.int4_result_u.data[1];
        *h = result.int4_result_u.data[2];
        *w = result.int4_result_u.data[3];
    }
    return result.err;
}

cudnnStatus_t cudnnDestroyPoolingDescriptor(cudnnPoolingDescriptor_t poolingDesc)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval_1;
    retval_1 = rpc_cudnndestroypoolingdescriptor_1(
        (ptr)poolingDesc,
        &result, clnt);

    if (retval_1 != RPC_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (%d)", __FUNCTION__, retval_1);
    }
    if (result != CUDNN_STATUS_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (result is %d)", __FUNCTION__, result);
    }
    return result;
}

DEF_FN(cudnnStatus_t, cudnnPoolingForward, cudnnHandle_t, handle, const cudnnPoolingDescriptor_t, poolingDesc, const void *, alpha, const cudnnTensorDescriptor_t, xDesc, const void *, x, const void *, beta, const cudnnTensorDescriptor_t, yDesc, void *, y)

cudnnStatus_t cudnnCreateActivationDescriptor(cudnnActivationDescriptor_t * activationDesc)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    ptr_result result;
    enum clnt_stat retval_1;
    if (activationDesc == NULL) {
        LOGE(LOG_ERROR, "%s failed (value is NULL)", __FUNCTION__);
        return CUDNN_STATUS_BAD_PARAM;
    }
    retval_1 = rpc_cudnncreateactivationdescriptor_1(&result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (%d)", __FUNCTION__, retval_1);
    }
    if (result.err != CUDNN_STATUS_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (result is %d)", __FUNCTION__, result.err);
    } else {
        *activationDesc = (cudnnActivationDescriptor_t)result.ptr_result_u.ptr;
    }
    return result.err;
}

cudnnStatus_t cudnnSetActivationDescriptor(cudnnActivationDescriptor_t activationDesc, cudnnActivationMode_t mode, cudnnNanPropagation_t reluNanOpt, double coef) 
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval_1;
    retval_1 = rpc_cudnnsetactivationdescriptor_1(
        (ptr)activationDesc,
        (int)mode,
        (int)reluNanOpt,
        coef,
        &result, clnt);

    if (retval_1 != RPC_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (%d)", __FUNCTION__, retval_1);
    }
    if (result != CUDNN_STATUS_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (result is %d)", __FUNCTION__, result);
    } 
    return result;
}

cudnnStatus_t cudnnGetActivationDescriptor(const cudnnActivationDescriptor_t activationDesc, cudnnActivationMode_t *mode, cudnnNanPropagation_t *reluNanOpt, double *coef) 
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int2d1_result result;
    enum clnt_stat retval_1;
    if (mode == NULL || reluNanOpt == NULL || coef == NULL) {
        LOGE(LOG_ERROR, "%s failed (value is NULL)", __FUNCTION__);
        return CUDNN_STATUS_BAD_PARAM;
    }
    retval_1 = rpc_cudnngetactivationdescriptor_1(
        (ptr)activationDesc,
        &result, clnt);

    if (retval_1 != RPC_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (%d)", __FUNCTION__, retval_1);
    }
    if (result.err != CUDNN_STATUS_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (result is %d)", __FUNCTION__, result);
    } else {
        *mode = (cudnnActivationMode_t)result.int2d1_result_u.data.i[0];
        *reluNanOpt = (cudnnNanPropagation_t)result.int2d1_result_u.data.i[1];
        *coef = result.int2d1_result_u.data.d;
    }
    return result.err;
}

cudnnStatus_t cudnnSetActivationDescriptorSwishBeta(cudnnActivationDescriptor_t activationDesc, double swish_beta)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval_1;
    retval_1 = rpc_cudnnsetactivationdescriptorswishbeta_1(
        (ptr)activationDesc,
        swish_beta,
        &result, clnt);

    if (retval_1 != RPC_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (%d)", __FUNCTION__, retval_1);
    }
    if (result != CUDNN_STATUS_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (result is %d)", __FUNCTION__, result);
    } 
    return result;
}
    
cudnnStatus_t cudnnGetActivationDescriptorSwishBeta(cudnnActivationDescriptor_t activationDesc, double * swish_beta)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    d_result result;
    enum clnt_stat retval_1;
    if (swish_beta == NULL) {
        LOGE(LOG_ERROR, "%s failed (value is NULL)", __FUNCTION__);
        return CUDNN_STATUS_BAD_PARAM;
    }
    retval_1 = rpc_cudnngetactivationdescriptorswishbeta_1(
        (ptr)activationDesc,
        &result, clnt);

    if (retval_1 != RPC_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (%d)", __FUNCTION__, retval_1);
    }
    if (result.err != CUDNN_STATUS_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (result is %d)", __FUNCTION__, result);
    } else {
        *swish_beta = result.d_result_u.data;
    }
    return result.err;
}

cudnnStatus_t cudnnDestroyActivationDescriptor(cudnnActivationDescriptor_t activationDesc)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval_1;
    retval_1 = rpc_cudnndestroyactivationdescriptor_1(
        (ptr)activationDesc,
        &result, clnt);

    if (retval_1 != RPC_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (%d)", __FUNCTION__, retval_1);
    }
    if (result != CUDNN_STATUS_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (result is %d)", __FUNCTION__, result);
    }
    return result;
}

DEF_FN(cudnnStatus_t, cudnnActivationForward, cudnnHandle_t, handle, cudnnActivationDescriptor_t, activationDesc, const void *, alpha, const cudnnTensorDescriptor_t, xDesc, const void *, x, const void *, beta, const cudnnTensorDescriptor_t, yDesc, void *, y)

cudnnStatus_t cudnnCreateLRNDescriptor(cudnnLRNDescriptor_t * normDesc)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    ptr_result result;
    enum clnt_stat retval_1;
    if (normDesc == NULL) {
        LOGE(LOG_ERROR, "%s failed (value is NULL)", __FUNCTION__);
        return CUDNN_STATUS_BAD_PARAM;
    }
    retval_1 = rpc_cudnncreatelrndescriptor_1(&result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (%d)", __FUNCTION__, retval_1);
    }
    if (result.err != CUDNN_STATUS_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (result is %d)", __FUNCTION__, result.err);
    } else {
        *normDesc = (cudnnLRNDescriptor_t)result.ptr_result_u.ptr;
    }
    return result.err;
}

cudnnStatus_t cudnnSetLRNDescriptor(cudnnLRNDescriptor_t normDesc, unsigned lrnN, double lrnAlpha, double lrnBeta, double lrnK)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval_1;
    retval_1 = rpc_cudnnsetlrndescriptor_1(
        (ptr)normDesc,
        (int)lrnN,
        lrnAlpha,
        lrnBeta,
        lrnK,
        &result, clnt);

    if (retval_1 != RPC_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (%d)", __FUNCTION__, retval_1);
    }
    if (result != CUDNN_STATUS_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (result is %d)", __FUNCTION__, result);
    } 
    return result;
}

cudnnStatus_t cudnnGetLRNDescriptor(cudnnLRNDescriptor_t normDesc, unsigned * lrnN, double * lrnAlpha, double * lrnBeta, double * lrnK)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int1d3_result result;
    enum clnt_stat retval_1;
    if (lrnN == NULL || lrnAlpha == NULL || lrnBeta == NULL || lrnK == NULL) {
        LOGE(LOG_ERROR, "%s failed (value is NULL)", __FUNCTION__);
        return CUDNN_STATUS_BAD_PARAM;
    }
    retval_1 = rpc_cudnngetlrndescriptor_1(
        (ptr)normDesc,
        &result, clnt);

    if (retval_1 != RPC_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (%d)", __FUNCTION__, retval_1);
    }
    if (result.err != CUDNN_STATUS_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (result is %d)", __FUNCTION__, result);
    } else {
        *lrnN = result.int1d3_result_u.data.i;
        *lrnAlpha = result.int1d3_result_u.data.d[0];
        *lrnBeta = result.int1d3_result_u.data.d[1];
        *lrnK = result.int1d3_result_u.data.d[2];
    }
    return result.err;
}

cudnnStatus_t cudnnDestroyLRNDescriptor(cudnnLRNDescriptor_t lrnDesc)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval_1;
    retval_1 = rpc_cudnndestroylrndescriptor_1(
        (ptr)lrnDesc,
        &result, clnt);

    if (retval_1 != RPC_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (%d)", __FUNCTION__, retval_1);
    }
    if (result != CUDNN_STATUS_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (result is %d)", __FUNCTION__, result);
    }
    return result;
}
DEF_FN(cudnnStatus_t, cudnnLRNCrossChannelForward, cudnnHandle_t, handle, cudnnLRNDescriptor_t, normDesc, cudnnLRNMode_t, lrnMode, const void *, alpha, const cudnnTensorDescriptor_t, xDesc, const void *, x, const void *, beta, const cudnnTensorDescriptor_t, yDesc, void *, y)
DEF_FN(cudnnStatus_t, cudnnDivisiveNormalizationForward, cudnnHandle_t, handle, cudnnLRNDescriptor_t, normDesc, cudnnDivNormMode_t, mode, const void *, alpha, const cudnnTensorDescriptor_t, xDesc, const void *, x, const void *, means, void *, temp, void *, temp2, const void *, beta, const cudnnTensorDescriptor_t, yDesc, void *, y)
DEF_FN(cudnnStatus_t, cudnnDeriveBNTensorDescriptor, cudnnTensorDescriptor_t, derivedBnDesc, const cudnnTensorDescriptor_t, xDesc, cudnnBatchNormMode_t, mode)
DEF_FN(cudnnStatus_t, cudnnBatchNormalizationForwardInference, cudnnHandle_t, handle, cudnnBatchNormMode_t, mode, const void *, alpha, const void *, beta, const cudnnTensorDescriptor_t, xDesc, const void *, x, const cudnnTensorDescriptor_t, yDesc, void *, y, const cudnnTensorDescriptor_t,  bnScaleBiasMeanVarDesc, const void *, bnScale, const void *, bnBias, const void *, estimatedMean, const void *, estimatedVariance, double, epsilon)
DEF_FN(cudnnStatus_t, cudnnDeriveNormTensorDescriptor, cudnnTensorDescriptor_t, derivedNormScaleBiasDesc, cudnnTensorDescriptor_t, derivedNormMeanVarDesc, const cudnnTensorDescriptor_t, xDesc, cudnnNormMode_t, mode, int, groupCnt) 
DEF_FN(cudnnStatus_t, cudnnNormalizationForwardInference, cudnnHandle_t, handle, cudnnNormMode_t, mode, cudnnNormOps_t, normOps, cudnnNormAlgo_t, algo, const void *, alpha, const void *, beta, const cudnnTensorDescriptor_t, xDesc, const void *, x, const cudnnTensorDescriptor_t normScaleBiasDesc, const void *, normScale, const void *, normBias, const cudnnTensorDescriptor_t, normMeanVarDesc, const void *, estimatedMean, const void *, estimatedVariance, const cudnnTensorDescriptor_t, zDesc, const void *, z, cudnnActivationDescriptor_t, activationDesc, const cudnnTensorDescriptor_t, yDesc, void *, y, double, epsilon, int, groupCnt) 
DEF_FN(cudnnStatus_t, cudnnCreateSpatialTransformerDescriptor, cudnnSpatialTransformerDescriptor_t *, stDesc)
DEF_FN(cudnnStatus_t, cudnnSetSpatialTransformerNdDescriptor, cudnnSpatialTransformerDescriptor_t, stDesc, cudnnSamplerType_t, samplerType, cudnnDataType_t, dataType, const int, nbDims, const int*, dimA)
DEF_FN(cudnnStatus_t, cudnnDestroySpatialTransformerDescriptor, cudnnSpatialTransformerDescriptor_t, stDesc)
DEF_FN(cudnnStatus_t, cudnnSpatialTfGridGeneratorForward, cudnnHandle_t, handle, const cudnnSpatialTransformerDescriptor_t, stDesc, const void *, theta, void *, grid)
DEF_FN(cudnnStatus_t, cudnnSpatialTfSamplerForward, cudnnHandle_t, handle, cudnnSpatialTransformerDescriptor_t, stDesc, const void *, alpha, const cudnnTensorDescriptor_t, xDesc, const void *, x, const void *, grid, const void *, beta, cudnnTensorDescriptor_t, yDesc, void *, y)
DEF_FN(cudnnStatus_t, cudnnCreateDropoutDescriptor, cudnnDropoutDescriptor_t *, dropoutDesc)
DEF_FN(cudnnStatus_t, cudnnDestroyDropoutDescriptor, cudnnDropoutDescriptor_t, dropoutDesc)
DEF_FN(cudnnStatus_t, cudnnDropoutGetStatesSize, cudnnHandle_t, handle, size_t *, sizeInBytes)
DEF_FN(cudnnStatus_t, cudnnDropoutGetReserveSpaceSize, cudnnTensorDescriptor_t, xdesc, size_t*, sizeInBytes)
DEF_FN(cudnnStatus_t, cudnnSetDropoutDescriptor, cudnnDropoutDescriptor_t, dropoutDesc, cudnnHandle_t, handle, float, dropout, void *, states, size_t, stateSizeInBytes, unsigned long long, seed)
DEF_FN(cudnnStatus_t, cudnnRestoreDropoutDescriptor, cudnnDropoutDescriptor_t, dropoutDesc, cudnnHandle_t, handle, float, dropout, void *, states, size_t, stateSizeInBytes, unsigned long long, seed)
DEF_FN(cudnnStatus_t, cudnnGetDropoutDescriptor, cudnnDropoutDescriptor_t, dropoutDesc, cudnnHandle_t, handle, float *, dropout, void **, states, unsigned long long *, seed)
DEF_FN(cudnnStatus_t, cudnnDropoutForward, cudnnHandle_t, handle, const cudnnDropoutDescriptor_t, dropoutDesc, const cudnnTensorDescriptor_t, xdesc, const void *, x, const cudnnTensorDescriptor_t, ydesc, void *, y, void *, reserveSpace, size_t, reserveSpaceSizeInBytes)
DEF_FN(cudnnStatus_t, cudnnCreateAlgorithmDescriptor, cudnnAlgorithmDescriptor_t *, algoDesc)
DEF_FN(cudnnStatus_t, cudnnSetAlgorithmDescriptor, cudnnAlgorithmDescriptor_t, algoDesc, cudnnAlgorithm_t, algorithm)
DEF_FN(cudnnStatus_t, cudnnGetAlgorithmDescriptor, const cudnnAlgorithmDescriptor_t, algoDesc, cudnnAlgorithm_t *, algorithm)
DEF_FN(cudnnStatus_t, cudnnCopyAlgorithmDescriptor, const cudnnAlgorithmDescriptor_t, src, cudnnAlgorithmDescriptor_t, dest)
DEF_FN(cudnnStatus_t, cudnnDestroyAlgorithmDescriptor, cudnnAlgorithmDescriptor_t, algoDesc)
DEF_FN(cudnnStatus_t, cudnnCreateAlgorithmPerformance, cudnnAlgorithmPerformance_t *, algoPerf, int, numberToCreate)
DEF_FN(cudnnStatus_t, cudnnSetAlgorithmPerformance, cudnnAlgorithmPerformance_t, algoPerf, cudnnAlgorithmDescriptor_t, algoDesc, cudnnStatus_t, status, float, time, size_t, memory)
DEF_FN(cudnnStatus_t, cudnnGetAlgorithmPerformance, const cudnnAlgorithmPerformance_t, algoPerf, cudnnAlgorithmDescriptor_t *, algoDesc, cudnnStatus_t, *, status, float *, time, size_t*, memory)
DEF_FN(cudnnStatus_t, cudnnDestroyAlgorithmPerformance, cudnnAlgorithmPerformance_t *, algoPerf, int, numberToDestroy)
DEF_FN(cudnnStatus_t, cudnnGetAlgorithmSpaceSize, cudnnHandle_t, handle, cudnnAlgorithmDescriptor_t, algoDesc, size_t *, algoSpaceSizeInBytes)
DEF_FN(cudnnStatus_t, cudnnSaveAlgorithm, cudnnHandle_t, handle, cudnnAlgorithmDescriptor_t, algoDesc, void *, algoSpace, size_t, algoSpaceSizeInBytes)
DEF_FN(cudnnStatus_t, cudnnRestoreAlgorithm, cudnnHandle_t, handle, void *, algoSpace, size_t, algoSpaceSizeInBytes, cudnnAlgorithmDescriptor_t, algoDesc)
DEF_FN(cudnnStatus_t, cudnnSetCallback, unsigned, mask, void *, udata, cudnnCallback_t, fptr)
DEF_FN(cudnnStatus_t, cudnnGetCallback, unsigned *, mask, void **, udata, cudnnCallback_t *, fptr)
DEF_FN(cudnnStatus_t, cudnnOpsInferVersionCheck)
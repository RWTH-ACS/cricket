#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <cublasLt.h>

//for strerror
#include <string.h>
#include <errno.h>

#include "cpu-libwrap.h"
#include "cpu_rpc_prot.h"
#include "cpu-common.h"
#include "cpu-utils.h"
#include "log.h"

#ifdef WITH_API_CNT
extern int api_call_cnt;
#endif //WITH_API_CNT

cublasStatus_t cublasLtCreate(cublasLtHandle_t *lighthandle)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    ptr_result result;
    enum clnt_stat retval_1;
    retval_1 = rpc_cublasltcreate_1(&result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    if (result.err == 0) {
        *lighthandle = (cublasLtHandle_t)result.ptr_result_u.ptr;
    }
    return result.err;
}

cublasStatus_t cublasLtMatrixLayoutCreate( cublasLtMatrixLayout_t *matLayout,
                                           cudaDataType type,
                                           uint64_t rows,
                                           uint64_t cols,
                                           int64_t ld)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT

    ptr_result result;
    enum clnt_stat retval_1;
    retval_1 = rpc_cublasltmatrixlayoutcreate_1(
        type,
        rows,
        cols,
        ld,
        &result,
        clnt
    );
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    if (result.err == 0) {
        *matLayout = (void*)result.ptr_result_u.ptr;
    }
    return result.err;
}

cublasStatus_t cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t *pref)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT

    ptr_result result;
    enum clnt_stat retval_1;
    retval_1 = rpc_cublasltmatmulpreferencecreate_1(&result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    if (result.err == 0) {
        *pref = (void*)result.ptr_result_u.ptr;
    }
    return result.err;
}


cublasStatus_t cublasLtMatmulDescCreate( cublasLtMatmulDesc_t *matmulDesc,
                                         cublasComputeType_t computeType,
                                         cudaDataType_t scaleType)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT

    ptr_result result;
    enum clnt_stat retval_1;
    retval_1 = rpc_cublasltmatmuldesccreate_1(
        computeType,
        scaleType,
        &result,
        clnt
    );

    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    if (result.err == 0) {
        *matmulDesc = (cublasLtMatmulDesc_t)result.ptr_result_u.ptr;
    }
    return result.err;
}

cublasStatus_t cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t matmulDesc)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT

    int result;
    enum clnt_stat retval_1;
    retval_1 = rpc_cublasltmatmuldescdestroy_1((ptr)matmulDesc, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    return result;
}


cublasStatus_t cublasLtMatmulAlgoGetHeuristic(
      cublasLtHandle_t lightHandle,
      cublasLtMatmulDesc_t operationDesc,
      cublasLtMatrixLayout_t Adesc,
      cublasLtMatrixLayout_t Bdesc,
      cublasLtMatrixLayout_t Cdesc,
      cublasLtMatrixLayout_t Ddesc,
      cublasLtMatmulPreference_t preference,
      int requestedAlgoCount,
      cublasLtMatmulHeuristicResult_t heuristicResultsArray[],
      int* returnAlgoCount)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT

    matmul_hr_result result;
    enum clnt_stat retval_1;
    retval_1 = rpc_cublasltmatmulalgogetheuristic_1(
        (ptr)lightHandle,
        (ptr)operationDesc,
        (ptr)Adesc,
        (ptr)Bdesc,
        (ptr)Cdesc,
        (ptr)Ddesc,
        (ptr)preference,
        requestedAlgoCount,
        &result,
        clnt
    );

    if (retval_1 != RPC_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (%d)", __FUNCTION__, retval_1);
    }
    if (result.err != 0) {
      return result.err;
    }

    *returnAlgoCount = result.matmul_hr_result_u.data.s;
    if (memcpy(heuristicResultsArray, result.matmul_hr_result_u.data.p, 96) == NULL) {
      LOGE(LOG_ERROR, "error: matmul hr alloc");
      return result.err;
    }

    return result.err;
}

cublasStatus_t cublasLtMatmulPreferenceSetAttribute(
      cublasLtMatmulPreference_t pref,
      cublasLtMatmulPreferenceAttributes_t attr,
      const void *buf,
      size_t sizeInBytes)
{
    return 0;
}

cublasStatus_t cublasLtMatmulPreferenceDestroy(cublasLtMatmulPreference_t pref)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT

    int result;
    enum clnt_stat retval_1;
    retval_1 = rpc_cublasltmatmulpreferencedestroy_1((ptr)pref, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    return result;
}

cublasStatus_t cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t matLayout)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT

    int result;
    enum clnt_stat retval_1;
    retval_1 = rpc_cublasltmatrixlayoutdestroy_1((ptr)matLayout, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    return result;
}

cublasStatus_t cublasLtMatmulDescSetAttribute(
      cublasLtMatmulDesc_t matmulDesc,
      cublasLtMatmulDescAttributes_t attr,
      const void *buf,
      size_t sizeInBytes)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT

    mem_data data = {
        .mem_data_len = sizeInBytes,
        .mem_data_val = (char *)buf
    };

    int result;
    enum clnt_stat retval_1;
    retval_1 = rpc_cublasltmatmuldescsetattribute_1(
        (ptr)matmulDesc,
        attr,
	data,
	&result,
	clnt
    );

    if (retval_1 != RPC_SUCCESS) {
        LOGE(LOG_ERROR, "%s failed (%d)", __FUNCTION__, retval_1);
    }

    return result;
}


cublasStatus_t cublasLtMatmul(
      cublasLtHandle_t               lightHandle,
      cublasLtMatmulDesc_t           computeDesc,
      const void                    *alpha,
      const void                    *A,
      cublasLtMatrixLayout_t         Adesc,
      const void                    *B,
      cublasLtMatrixLayout_t         Bdesc,
      const void                    *beta,
      const void                    *C,
      cublasLtMatrixLayout_t         Cdesc,
      void                          *D,
      cublasLtMatrixLayout_t         Ddesc,
      const cublasLtMatmulAlgo_t    *algo,
      void                          *workspace,
      size_t                         workspaceSizeInBytes,
      cudaStream_t                   stream)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT

    int result;
    enum clnt_stat retval_1;
    retval_1 = rpc_cublasltmatmul_1(
      (ptr)lightHandle,
      (ptr)computeDesc,
      *((float*)alpha),
      (ptr)A,
      (ptr)Adesc,
      (ptr)B,
      (ptr)Bdesc,
      *((float*)beta),
      (ptr)C,
      (ptr)Cdesc,
      (ptr)D,
      (ptr)Ddesc,
      (ptr)algo,
      (ptr)workspace,
      workspaceSizeInBytes,
      (ptr)stream,
      &result,
      clnt
    );
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    return result;
}


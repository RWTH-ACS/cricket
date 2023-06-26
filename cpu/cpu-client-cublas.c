
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cublas_v2.h>

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

cublasStatus_t cublasCreate(cublasHandle_t* handle)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    ptr_result result;
    enum clnt_stat retval_1;
    retval_1 = rpc_cublascreate_1(&result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    if (result.err == 0) {
        *handle = (void*)result.ptr_result_u.ptr;
    }
    return result.err;
}

cublasStatus_t cublasDestroy(cublasHandle_t handle)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval_1;
    retval_1 = rpc_cublasdestroy_1((ptr)handle, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    return result;
}

cublasStatus_t cublasDgemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const double          *alpha,
                           const double          *A, int lda,
                           const double          *B, int ldb,
                           const double          *beta,
                           double          *C, int ldc)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval_1;
    retval_1 = rpc_cublasdgemm_1(
        (ptr)handle,
        (int)transa,
        (int)transb,
        m, n, k,
        *alpha,
        (ptr)A, lda,
        (ptr)B, ldb,
        *beta,
        (ptr)C, ldc,
         &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    return result;
}

cublasStatus_t cublasSgemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const float *alpha,
                           const float *A, int lda,
                           const float *B, int ldb,
                           const float *beta,
                           float *C, int ldc)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval_1;
    retval_1 = rpc_cublassgemm_1(
        (ptr)handle,
        (int)transa,
        (int)transb,
        m, n, k,
        *alpha,
        (ptr)A, lda,
        (ptr)B, ldb,
        *beta,
        (ptr)C, ldc,
         &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    return result;
}

cublasStatus_t cublasSgemmEx(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const float *alpha,
                           const void *A, cudaDataType_t Atype, int lda,
                           const void *B, cudaDataType_t Btype, int ldb,
                           const float *beta,
                           void *C, cudaDataType_t Ctype, int ldc)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval_1;
    retval_1 = rpc_cublassgemmex_1(
        (ptr)handle,
        (int)transa,
        (int)transb,
        m, n, k,
        *alpha,
        (ptr)A, (int)Atype, lda,
        (ptr)B, (int)Btype, ldb,
        *beta,
        (ptr)C, (int)Ctype, ldc,
         &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    return result;
}

cublasStatus_t cublasDgemv(cublasHandle_t handle,
                           cublasOperation_t trans,
                           int m, int n,
                           const double          *alpha,
                           const double          *A, int lda,
                           const double          *x, int incx,
                           const double          *beta,
                           double          *y, int incy)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval_1;
    retval_1 = rpc_cublasdgemv_1(
        (ptr)handle,
        (int)trans,
        m, n,
        *alpha,
        (ptr)A, lda,
        (ptr)x, incx,
        *beta,
        (ptr)y, incy,
         &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    return result;
}

cublasStatus_t cublasSgemv(cublasHandle_t handle,
                           cublasOperation_t trans,
                           int m, int n,
                           const float          *alpha,
                           const float          *A, int lda,
                           const float          *x, int incx,
                           const float          *beta,
                           float          *y, int incy)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval_1;
    retval_1 = rpc_cublassgemv_1(
        (ptr)handle,
        (int)trans,
        m, n,
        *alpha,
        (ptr)A, lda,
        (ptr)x, incx,
        *beta,
        (ptr)y, incy,
         &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    return result;
}
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

//for strerror
#include <string.h>
#include <errno.h>

#include "cpu_rpc_prot.h"
#include "cpu-common.h"
#include "cpu-utils.h"
#include "log.h"
#include "resource-mg.h"
#define WITH_RECORDER
#include "api-recorder.h"
#include "cpu-server-cublas.h"
#include "gsched.h"



int cublas_init(int bypass, resource_mg *memory)
{
    int ret = 0;
    ret &= resource_mg_init(&rm_cublas, bypass);
    return ret;
}

resource_mg *cublas_get_rm(void)
{
    return &rm_cublas;
}

int cublas_deinit(void)
{
    resource_mg_free(&rm_cublas);
    return 0;

}

bool_t rpc_cublascreate_1_svc(ptr_result *result, struct svc_req *rqstp)
{
    RECORD_VOID_API;
    LOGE(LOG_DEBUG, "cublasCreate_v2");

    GSCHED_RETAIN;
    result->err = cublasCreate_v2((cublasHandle_t*)&result->ptr_result_u.ptr);
    if (resource_mg_create(&rm_cublas, (void*)result->ptr_result_u.ptr) != 0) {
      LOGE(LOG_ERROR, "error in resource manager");
    }
    GSCHED_RELEASE;
    
    RECORD_RESULT(ptr_result_u, *result);
    return 1;
}

bool_t rpc_cublasdgemm_1_svc(ptr handle, int transa, int transb, int m, int n, int k, double alpha,
            ptr A, int lda,
            ptr B, int ldb, double beta,
            ptr C, int ldc,
            int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cublasdgemm_1_argument);
    RECORD_ARG(1, handle);
    RECORD_ARG(2, transa);
    RECORD_ARG(3, transb);
    RECORD_ARG(4, m);
    RECORD_ARG(5, n);
    RECORD_ARG(6, k);
    RECORD_ARG(7, alpha);
    RECORD_ARG(8, A);
    RECORD_ARG(9, lda);
    RECORD_ARG(10, B);
    RECORD_ARG(11, ldb);
    RECORD_ARG(12, beta);
    RECORD_ARG(13, C);
    RECORD_ARG(14, ldc);
    LOGE(LOG_DEBUG, "cublasDgemm(%p, %d, %d, %d, %d, %d, %d, %f, %p, %d, %p, %d, %f, %p, %d)",
        handle, transa, transb, m, n, k, lda, alpha, A, lda, B, ldb, beta, C, ldc);
    GSCHED_RETAIN;
    *result = cublasDgemm(resource_mg_get(&rm_cublas, (void*)handle),
                    (cublasOperation_t) transa,
                    (cublasOperation_t) transb,
                    m, n, k, &alpha,
                    resource_mg_get(&rm_memory, (void*)A), lda,
                    resource_mg_get(&rm_memory, (void*)B), ldb, &beta,
                    resource_mg_get(&rm_memory, (void*)C), ldc
    );
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cublasdestroy_1_svc(ptr handle, int *result, struct svc_req *rqstp)
{
    RECORD_API(ptr);
    RECORD_SINGLE_ARG(handle);
    LOGE(LOG_DEBUG, "cublasDestroy_v2");
    GSCHED_RETAIN;
    *result = cublasDestroy_v2(resource_mg_get(&rm_cublas, (void*)handle));
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cublassetworkspace_1_svc(ptr handle, ptr workspace, size_t workspaceSizeInBytes, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cublassetworkspace_1_argument);
    RECORD_NARG(handle);
    RECORD_NARG(workspace);
    RECORD_NARG(workspaceSizeInBytes);
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
#if CUBLAS_VERSION >= 11000
    *result = cublasSetWorkspace(
        resource_mg_get(&rm_cublas, (void*)handle),
        resource_mg_get(&rm_memory, (void*)workspace),
        workspaceSizeInBytes);
#else
    LOGE(LOG_ERROR, "cublassetworkspace not supported in this version");
    *result = -1;
#endif
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cublassetstream_1_svc(ptr handle, ptr streamId, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cublassetstream_1_argument);
    RECORD_NARG(handle);
    RECORD_NARG(streamId);
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    *result = cublasSetStream(
        resource_mg_get(&rm_cublas, (void*)handle),
        resource_mg_get(&rm_streams, (void*)streamId));
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cublassetmathmode_1_svc(ptr handle, int mode, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cublassetmathmode_1_argument);
    RECORD_NARG(handle);
    RECORD_NARG(mode);
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    *result = cublasSetMathMode(
        resource_mg_get(&rm_cublas, (void*)handle),
        (cublasMath_t)mode);
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cublassgemm_1_svc(ptr handle, int transa, int transb, int m, int n, int k, float alpha,
            ptr A, int lda,
            ptr B, int ldb, float beta,
            ptr C, int ldc,
            int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cublassgemm_1_argument);
    RECORD_ARG(1, handle);
    RECORD_ARG(2, transa);
    RECORD_ARG(3, transb);
    RECORD_ARG(4, m);
    RECORD_ARG(5, n);
    RECORD_ARG(6, k);
    RECORD_ARG(7, alpha);
    RECORD_ARG(8, A);
    RECORD_ARG(9, lda);
    RECORD_ARG(10, B);
    RECORD_ARG(11, ldb);
    RECORD_ARG(12, beta);
    RECORD_ARG(13, C);
    RECORD_ARG(14, ldc);
    LOGE(LOG_DEBUG, "cublasSgemm");
    GSCHED_RETAIN;
#if CUBLAS_VERSION >= 11000
    *result = cublasSgemm(resource_mg_get(&rm_cublas, (void*)handle),
                    (cublasOperation_t) transa,
                    (cublasOperation_t) transb,
                    m, n, k, &alpha,
                    resource_mg_get(&rm_memory, (void*)A), lda,
                    resource_mg_get(&rm_memory, (void*)B), ldb, &beta,
                    resource_mg_get(&rm_memory, (void*)C), ldc
    );
#else
    LOGE(LOG_ERROR, "cublassetworkspace not supported in this version");
    *result = -1;
#endif
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cublassgemv_1_svc(ptr handle, int trans, int m, 
            int n, float alpha,
            ptr A, int lda,
            ptr x, int incx, float beta,
            ptr y, int incy,
            int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cublassgemv_1_argument);
    RECORD_ARG(1, handle);
    RECORD_ARG(2, trans);
    RECORD_ARG(3, m);
    RECORD_ARG(4, n);
    RECORD_ARG(5, alpha);
    RECORD_ARG(6, A);
    RECORD_ARG(7, lda);
    RECORD_ARG(8, x);
    RECORD_ARG(9, incx);
    RECORD_ARG(10, beta);
    RECORD_ARG(11, y);
    RECORD_ARG(12, incy);
    LOGE(LOG_DEBUG, "cublasSgemv");
    GSCHED_RETAIN;
    *result = cublasSgemv(resource_mg_get(&rm_cublas, (void*)handle),
                    (cublasOperation_t) trans,
                    m, n, &alpha,
                    resource_mg_get(&rm_memory, (void*)A), lda,
                    resource_mg_get(&rm_memory, (void*)x), incx, &beta,
                    resource_mg_get(&rm_memory, (void*)y), incy
    );
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cublasdgemv_1_svc(ptr handle, int trans, int m, 
            int n, double alpha,
            ptr A, int lda,
            ptr x, int incx, double beta,
            ptr y, int incy,
            int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cublasdgemv_1_argument);
    RECORD_ARG(1, handle);
    RECORD_ARG(2, trans);
    RECORD_ARG(3, m);
    RECORD_ARG(4, n);
    RECORD_ARG(5, alpha);
    RECORD_ARG(6, A);
    RECORD_ARG(7, lda);
    RECORD_ARG(8, x);
    RECORD_ARG(9, incx);
    RECORD_ARG(10, beta);
    RECORD_ARG(11, y);
    RECORD_ARG(12, incy);
    LOGE(LOG_DEBUG, "cublasDgemv");
    GSCHED_RETAIN;
    *result = cublasDgemv(resource_mg_get(&rm_cublas, (void*)handle),
                    (cublasOperation_t) trans,
                    m, n, &alpha,
                    resource_mg_get(&rm_memory, (void*)A), lda,
                    resource_mg_get(&rm_memory, (void*)x), incx, &beta,
                    resource_mg_get(&rm_memory, (void*)y), incy
    );
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cublassgemmex_1_svc(ptr handle, int transa, int transb, int m, int n, int k, float alpha,
            ptr A, int Atype, int lda,
            ptr B, int Btype, int ldb, float beta,
            ptr C, int Ctype, int ldc,
            int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cublassgemmex_1_argument);
    RECORD_ARG(1, handle);
    RECORD_ARG(2, transa);
    RECORD_ARG(3, transb);
    RECORD_ARG(4, m);
    RECORD_ARG(5, n);
    RECORD_ARG(6, k);
    RECORD_ARG(7, alpha);
    RECORD_ARG(8, A);
    RECORD_ARG(9, Atype);
    RECORD_ARG(10, lda);
    RECORD_ARG(11, B);
    RECORD_ARG(12, Btype);
    RECORD_ARG(13, ldb);
    RECORD_ARG(14, beta);
    RECORD_ARG(15, C);
    RECORD_ARG(16, Ctype);
    RECORD_ARG(17, ldc);
    LOGE(LOG_DEBUG, "cublasSgemmEx");
    GSCHED_RETAIN;
    *result = cublasSgemmEx(resource_mg_get(&rm_cublas, (void*)handle),
                    (cublasOperation_t) transa,
                    (cublasOperation_t) transb,
                    m, n, k, &alpha,
                    resource_mg_get(&rm_memory, (void*)A), (cudaDataType_t)Atype, lda,
                    resource_mg_get(&rm_memory, (void*)B), (cudaDataType_t)Btype, ldb, &beta,
                    resource_mg_get(&rm_memory, (void*)C), (cudaDataType_t)Ctype, ldc
    );
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cublasgetmathmode_1_svc(ptr handle, int_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    result-> err = cublasGetMathMode(
        (cublasHandle_t)resource_mg_get(&rm_cublas, (void*)handle),
        (cublasMath_t*)&result->int_result_u.data
    );
    GSCHED_RELEASE;
    return 1;
}


bool_t rpc_cublasgemmstridedbatchedex_1_svc(
    ptr handle,
    int transa,
    int transb,
    int m, int n, int k,
    float alpha,
    ptr A,
    int Atype,
    int lda,
    ll strideA,
    ptr B,
    int Btype,
    int ldb,
    ll strideB,
    float beta,
    ptr C,
    int Ctype,
    int ldc,
    ll strideC,
    int batchCount,
    int computeType,
    int algo,
    int *result, struct svc_req *rqstp
)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    *result = cublasGemmStridedBatchedEx(
        (cublasHandle_t)resource_mg_get(&rm_cublas, (void*)handle),
        (cublasOperation_t) transa,
        (cublasOperation_t) transb,
        m, n, k, &alpha,
        resource_mg_get(&rm_memory, (void*)A), (cudaDataType_t)Atype, lda, (long long int)strideA,
        resource_mg_get(&rm_memory, (void*)B), (cudaDataType_t)Btype, ldb, (long long int)strideB,
        &beta,
        resource_mg_get(&rm_memory, (void*)C), (cudaDataType_t)Ctype, ldc, (long long int)strideC,
        batchCount,
        (cublasComputeType_t)computeType,
        (cublasGemmAlgo_t)algo
    );
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cublasgemmex_1_svc(ptr handle, int transa, int transb, int m, int n, int k, float alpha,
            ptr A, int Atype, int lda,
            ptr B, int Btype, int ldb, float beta,
            ptr C, int Ctype, int ldc,
            int computeType, int algo,
            int *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cublasGemmEx");
    GSCHED_RETAIN;
    *result = cublasGemmEx(resource_mg_get(&rm_cublas, (void*)handle),
                    (cublasOperation_t) transa,
                    (cublasOperation_t) transb,
                    m, n, k, &alpha,
                    resource_mg_get(&rm_memory, (void*)A), (cudaDataType_t)Atype, lda,
                    resource_mg_get(&rm_memory, (void*)B), (cudaDataType_t)Btype, ldb, &beta,
                    resource_mg_get(&rm_memory, (void*)C), (cudaDataType_t)Ctype, ldc,
        (cublasComputeType_t)computeType,
        (cublasGemmAlgo_t)algo
    );
    GSCHED_RELEASE;
    return 1;
}


bool_t rpc_cublasgemmstridedbatched_1_svc(
    ptr handle,
    int transa,
    int transb,
    int m, int n, int k,
    float alpha,
    ptr A,
    int lda,
    ll strideA,
    ptr B,
    int ldb,
    ll strideB,
    float beta,
    ptr C,
    int ldc,
    ll strideC,
    int batchCount,
    int *result, struct svc_req *rqstp
)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    *result = cublasSgemmStridedBatched(
        (cublasHandle_t)resource_mg_get(&rm_cublas, (void*)handle),
        (cublasOperation_t) transa,
        (cublasOperation_t) transb,
        m, n, k, &alpha,
        resource_mg_get(&rm_memory, (void*)A), lda, (long long int)strideA,
        resource_mg_get(&rm_memory, (void*)B), ldb, (long long int)strideB,
        &beta,
        resource_mg_get(&rm_memory, (void*)C), ldc, (long long int)strideC,
        batchCount
    );
    GSCHED_RELEASE;
    return 1;
}

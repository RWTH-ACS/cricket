#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>

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
extern size_t memcpy_cnt;
#endif //WITH_API_CNT

cusolverStatus_t cusolverDnCreate(cusolverDnHandle_t* handle)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    ptr_result result;
    enum clnt_stat retval_1;
    retval_1 = rpc_cusolverdncreate_1(&result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    if (result.err == 0) {
        *handle = (void*)result.ptr_result_u.ptr;
    }
    return result.err;
}
cusolverStatus_t cusolverDnDestroy(cusolverDnHandle_t handle)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval_1;
    retval_1 = rpc_cusolverdndestroy_1((ptr)handle, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    return result;
}
cusolverStatus_t cusolverDnSetStream(cusolverDnHandle_t handle, cudaStream_t streamId)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval_1;
    retval_1 = rpc_cusolverdnsetstream_1((ptr)handle, (ptr)streamId, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    return result;
}
DEF_FN(cusolverStatus_t, cusolverDnGetStream, cusolverDnHandle_t, handle, cudaStream_t*, streamId)

DEF_FN(cusolverStatus_t,
    cusolverDnIRSParamsCreate,
            cusolverDnIRSParams_t*, params_ptr )

DEF_FN(cusolverStatus_t,
    cusolverDnIRSParamsDestroy,
            cusolverDnIRSParams_t, params )

#if CUDART_VERSION >= 11000
DEF_FN(cusolverStatus_t,
    cusolverDnIRSParamsSetTol,
            cusolverDnIRSParams_t, params,
            double, val )
#else
DEF_FN(cusolverStatus_t,
    cusolverDnIRSParamsSetTol,
            cusolverDnIRSParams_t, params,
            cudaDataType, data_type,
            double, val )
#endif

#if CUDART_VERSION >= 11000
DEF_FN(cusolverStatus_t,
    cusolverDnIRSParamsSetTolInner,
            cusolverDnIRSParams_t, params,
            double, val )
#else
DEF_FN(cusolverStatus_t,
    cusolverDnIRSParamsSetTolInner,
            cusolverDnIRSParams_t, params,
            cudaDataType, data_type,
            double, val )
#endif

#if CUDART_VERSION >= 11000
DEF_FN(cusolverStatus_t,
    cusolverDnIRSParamsSetSolverPrecisions,
            cusolverDnIRSParams_t, params,
            cusolverPrecType_t, solver_main_precision,
            cusolverPrecType_t, solver_lowest_precision)
#else
DEF_FN(cusolverStatus_t,
    cusolverDnIRSParamsSetSolverPrecisions,
            cusolverDnIRSParams_t, params,
            cudaDataType, solver_main_precision,
            cudaDataType, solver_lowest_precision )
#endif

DEF_FN(cusolverStatus_t,
    cusolverDnIRSParamsSetRefinementSolver,
            cusolverDnIRSParams_t, params,
            cusolverIRSRefinement_t, refinement_solver )

DEF_FN(cusolverStatus_t,
    cusolverDnIRSParamsSetMaxIters,
            cusolverDnIRSParams_t, params,
            cusolver_int_t, maxiters )

DEF_FN(cusolverStatus_t,
    cusolverDnIRSParamsSetMaxItersInner,
            cusolverDnIRSParams_t, params,
            cusolver_int_t, maxiters_inner )

DEF_FN(cusolverStatus_t,
    cusolverDnIRSParamsGetNiters,
            cusolverDnIRSParams_t, params,
            cusolver_int_t*, niters )

DEF_FN(cusolverStatus_t,
    cusolverDnIRSParamsGetOuterNiters,
            cusolverDnIRSParams_t, params,
            cusolver_int_t*, outer_niters )

DEF_FN(cusolverStatus_t,
    cusolverDnIRSParamsGetMaxIters,
            cusolverDnIRSParams_t, params,
            cusolver_int_t*, maxiters )

#if CUDART_VERSION >= 11000
DEF_FN(cusolverStatus_t,
    cusolverDnIRSParamsSetSolverMainPrecision,
            cusolverDnIRSParams_t, params,
            cusolverPrecType_t, solver_main_precision )
#else
DEF_FN(cusolverStatus_t,
    cusolverDnIRSParamsSetSolverMainPrecision,
            cusolverDnIRSParams_t, params,
            cudaDataType, solver_main_precision )
#endif

#if CUDART_VERSION >= 11000
DEF_FN(cusolverStatus_t,
    cusolverDnIRSParamsSetSolverLowestPrecision,
            cusolverDnIRSParams_t, params,
            cusolverPrecType_t, lowest_precision_type )
#else
DEF_FN(cusolverStatus_t,
    cusolverDnIRSParamsSetSolverLowestPrecision,
            cusolverDnIRSParams_t, params,
            cudaDataType, solver_lowest_precision )
#endif

#if CUDART_VERSION >= 11000
DEF_FN(cusolverStatus_t, cusolverDnIRSInfosDestroy,
        cusolverDnIRSInfos_t, infos )
#else
DEF_FN(cusolverStatus_t, cusolverDnIRSInfosDestroy,
        cusolverDnIRSParams_t, params,
        cusolverDnIRSInfos_t, infos )
#endif

#if CUDART_VERSION >= 11000
DEF_FN(cusolverStatus_t, cusolverDnIRSInfosCreate,
        cusolverDnIRSInfos_t*, infos_ptr )
#else
DEF_FN(cusolverStatus_t, cusolverDnIRSInfosCreate,
        cusolverDnIRSParams_t, params,
        cusolverDnIRSInfos_t*, infos_ptr )
#endif

#if CUDART_VERSION >= 11000
DEF_FN(cusolverStatus_t,
    cusolverDnIRSInfosGetNiters,
            cusolverDnIRSInfos_t, infos,
            cusolver_int_t*, niters )
#else
DEF_FN(cusolverStatus_t,
    cusolverDnIRSInfosGetNiters,
            cusolverDnIRSParams_t, params,
            cusolverDnIRSInfos_t, infos,
            cusolver_int_t*, niters )
#endif

#if CUDART_VERSION >= 11000
DEF_FN(cusolverStatus_t,
    cusolverDnIRSInfosGetOuterNiters,
            cusolverDnIRSInfos_t, infos,
            cusolver_int_t*, outer_niters )
#else
DEF_FN(cusolverStatus_t,
    cusolverDnIRSInfosGetOuterNiters,
            cusolverDnIRSParams_t, params,
            cusolverDnIRSInfos_t, infos,
            cusolver_int_t*, outer_niters )
#endif

#if CUDART_VERSION >= 11000
DEF_FN(cusolverStatus_t,
    cusolverDnIRSInfosGetMaxIters,
            cusolverDnIRSInfos_t, infos,
            cusolver_int_t*, maxiters )
#else
DEF_FN(cusolverStatus_t,
    cusolverDnIRSInfosGetMaxIters,
            cusolverDnIRSParams_t, params,
            cusolverDnIRSInfos_t, infos,
            cusolver_int_t*, maxiters )
#endif

#if CUDART_VERSION >= 11000
DEF_FN(cusolverStatus_t, cusolverDnIRSInfosRequestResidual,
        cusolverDnIRSInfos_t, infos )
#else
DEF_FN(cusolverStatus_t, cusolverDnIRSInfosRequestResidual,
        cusolverDnIRSParams_t, params,
        cusolverDnIRSInfos_t, infos )
#endif

#if CUDART_VERSION >= 11000
DEF_FN(cusolverStatus_t, cusolverDnIRSInfosGetResidualHistory,
            cusolverDnIRSInfos_t, infos,
            void**, residual_history )
#else
DEF_FN(cusolverStatus_t, cusolverDnIRSInfosGetResidualHistory,
            cusolverDnIRSParams_t, params,
            cusolverDnIRSInfos_t, infos,
            void**, residual_history )
#endif

DEF_FN(cusolverStatus_t, cusolverDnZZgesv,
        cusolverDnHandle_t, handle,
        cusolver_int_t, n, cusolver_int_t, nrhs,
        cuDoubleComplex*, dA, cusolver_int_t, ldda,
        cusolver_int_t*, dipiv,
        cuDoubleComplex*, dB, cusolver_int_t, lddb,
        cuDoubleComplex*, dX, cusolver_int_t, lddx,
        void*, dWorkspace, size_t, lwork_bytes,
        cusolver_int_t*, iter,
        cusolver_int_t*, d_info)

DEF_FN(cusolverStatus_t, cusolverDnZCgesv,
        cusolverDnHandle_t, handle,
        cusolver_int_t, n, cusolver_int_t, nrhs,
        cuDoubleComplex*, dA, cusolver_int_t, ldda,
        cusolver_int_t*, dipiv,
        cuDoubleComplex*, dB, cusolver_int_t, lddb,
        cuDoubleComplex*, dX, cusolver_int_t, lddx,
        void*, dWorkspace, size_t, lwork_bytes,
        cusolver_int_t*, iter,
        cusolver_int_t*, d_info)

DEF_FN(cusolverStatus_t, cusolverDnZKgesv,
        cusolverDnHandle_t, handle,
        cusolver_int_t, n, cusolver_int_t, nrhs,
        cuDoubleComplex*, dA, cusolver_int_t, ldda,
        cusolver_int_t*, dipiv,
        cuDoubleComplex*, dB, cusolver_int_t, lddb,
        cuDoubleComplex*, dX, cusolver_int_t, lddx,
        void*, dWorkspace, size_t, lwork_bytes,
        cusolver_int_t*, iter,
        cusolver_int_t*, d_info)

DEF_FN(cusolverStatus_t, cusolverDnCCgesv,
        cusolverDnHandle_t, handle,
        cusolver_int_t, n, cusolver_int_t, nrhs,
        cuComplex*, dA, cusolver_int_t, ldda,
        cusolver_int_t*, dipiv,
        cuComplex*, dB, cusolver_int_t, lddb,
        cuComplex*, dX, cusolver_int_t, lddx,
        void*, dWorkspace, size_t, lwork_bytes,
        cusolver_int_t*, iter,
        cusolver_int_t*, d_info)

DEF_FN(cusolverStatus_t, cusolverDnCKgesv,
        cusolverDnHandle_t, handle,
        cusolver_int_t, n, cusolver_int_t, nrhs,
        cuComplex*, dA, cusolver_int_t, ldda,
        cusolver_int_t*, dipiv,
        cuComplex*, dB, cusolver_int_t, lddb,
        cuComplex*, dX, cusolver_int_t, lddx,
        void*, dWorkspace, size_t, lwork_bytes,
        cusolver_int_t*, iter,
        cusolver_int_t*, d_info)

DEF_FN(cusolverStatus_t, cusolverDnDDgesv,
        cusolverDnHandle_t, handle,
        cusolver_int_t, n, cusolver_int_t, nrhs,
        double*, dA, cusolver_int_t, ldda,
        cusolver_int_t*, dipiv,
        double*, dB, cusolver_int_t, lddb,
        double*, dX, cusolver_int_t, lddx,
        void*, dWorkspace, size_t, lwork_bytes,
        cusolver_int_t*, iter,
        cusolver_int_t*, d_info)

DEF_FN(cusolverStatus_t, cusolverDnDSgesv,
        cusolverDnHandle_t, handle,
        cusolver_int_t, n, cusolver_int_t, nrhs,
        double*, dA, cusolver_int_t, ldda,
        cusolver_int_t*, dipiv,
        double*, dB, cusolver_int_t, lddb,
        double*, dX, cusolver_int_t, lddx,
        void*, dWorkspace, size_t, lwork_bytes,
        cusolver_int_t*, iter,
        cusolver_int_t*, d_info)

DEF_FN(cusolverStatus_t, cusolverDnDHgesv,
        cusolverDnHandle_t, handle,
        cusolver_int_t, n, cusolver_int_t, nrhs,
        double*, dA, cusolver_int_t, ldda,
        cusolver_int_t*, dipiv,
        double*, dB, cusolver_int_t, lddb,
        double*, dX, cusolver_int_t, lddx,
        void*, dWorkspace, size_t, lwork_bytes,
        cusolver_int_t*, iter,
        cusolver_int_t*, d_info)

DEF_FN(cusolverStatus_t, cusolverDnSSgesv,
        cusolverDnHandle_t, handle,
        cusolver_int_t, n, cusolver_int_t, nrhs,
        float*, dA, cusolver_int_t, ldda,
        cusolver_int_t*, dipiv,
        float*, dB, cusolver_int_t, lddb,
        float*, dX, cusolver_int_t, lddx,
        void*, dWorkspace, size_t, lwork_bytes,
        cusolver_int_t*, iter,
        cusolver_int_t*, d_info)

DEF_FN(cusolverStatus_t, cusolverDnSHgesv,
        cusolverDnHandle_t, handle,
        cusolver_int_t, n, cusolver_int_t, nrhs,
        float*, dA, cusolver_int_t, ldda,
        cusolver_int_t*, dipiv,
        float*, dB, cusolver_int_t, lddb,
        float*, dX, cusolver_int_t, lddx,
        void*, dWorkspace, size_t, lwork_bytes,
        cusolver_int_t*, iter,
        cusolver_int_t*, d_info)

DEF_FN(cusolverStatus_t, cusolverDnZZgesv_bufferSize,
        cusolverDnHandle_t, handle,
        cusolver_int_t, n, cusolver_int_t, nrhs,
        cuDoubleComplex*, dA, cusolver_int_t, ldda,
        cusolver_int_t*, dipiv,
        cuDoubleComplex*, dB, cusolver_int_t, lddb,
        cuDoubleComplex*, dX, cusolver_int_t, lddx,
        void*, dWorkspace, size_t*, lwork_bytes)

DEF_FN(cusolverStatus_t, cusolverDnZCgesv_bufferSize,
        cusolverDnHandle_t, handle,
        cusolver_int_t, n, cusolver_int_t, nrhs,
        cuDoubleComplex*, dA, cusolver_int_t, ldda,
        cusolver_int_t*, dipiv,
        cuDoubleComplex*, dB, cusolver_int_t, lddb,
        cuDoubleComplex*, dX, cusolver_int_t, lddx,
        void*, dWorkspace, size_t*, lwork_bytes)

DEF_FN(cusolverStatus_t, cusolverDnZKgesv_bufferSize,
        cusolverDnHandle_t, handle,
        cusolver_int_t, n, cusolver_int_t, nrhs,
        cuDoubleComplex*, dA, cusolver_int_t, ldda,
        cusolver_int_t*, dipiv,
        cuDoubleComplex*, dB, cusolver_int_t, lddb,
        cuDoubleComplex*, dX, cusolver_int_t, lddx,
        void*, dWorkspace, size_t*, lwork_bytes)

DEF_FN(cusolverStatus_t, cusolverDnCCgesv_bufferSize,
        cusolverDnHandle_t, handle,
        cusolver_int_t, n, cusolver_int_t, nrhs,
        cuComplex*, dA, cusolver_int_t, ldda,
        cusolver_int_t*, dipiv,
        cuComplex*, dB, cusolver_int_t, lddb,
        cuComplex*, dX, cusolver_int_t, lddx,
        void*, dWorkspace, size_t*, lwork_bytes)

DEF_FN(cusolverStatus_t, cusolverDnCKgesv_bufferSize,
        cusolverDnHandle_t, handle,
        cusolver_int_t, n, cusolver_int_t, nrhs,
        cuComplex*, dA, cusolver_int_t, ldda,
        cusolver_int_t*, dipiv,
        cuComplex*, dB, cusolver_int_t, lddb,
        cuComplex*, dX, cusolver_int_t, lddx,
        void*, dWorkspace, size_t*, lwork_bytes)

DEF_FN(cusolverStatus_t, cusolverDnDDgesv_bufferSize,
        cusolverDnHandle_t, handle,
        cusolver_int_t, n, cusolver_int_t, nrhs,
        double*, dA, cusolver_int_t, ldda,
        cusolver_int_t*, dipiv,
        double*, dB, cusolver_int_t, lddb,
        double*, dX, cusolver_int_t, lddx,
        void*, dWorkspace, size_t*, lwork_bytes)

DEF_FN(cusolverStatus_t, cusolverDnDSgesv_bufferSize,
        cusolverDnHandle_t, handle,
        cusolver_int_t, n, cusolver_int_t, nrhs,
        double*, dA, cusolver_int_t, ldda,
        cusolver_int_t*, dipiv,
        double*, dB, cusolver_int_t, lddb,
        double*, dX, cusolver_int_t, lddx,
        void*, dWorkspace, size_t*, lwork_bytes)

DEF_FN(cusolverStatus_t, cusolverDnDHgesv_bufferSize,
        cusolverDnHandle_t, handle,
        cusolver_int_t, n, cusolver_int_t, nrhs,
        double*, dA, cusolver_int_t, ldda,
        cusolver_int_t*, dipiv,
        double*, dB, cusolver_int_t, lddb,
        double*, dX, cusolver_int_t, lddx,
        void*, dWorkspace, size_t*, lwork_bytes)

DEF_FN(cusolverStatus_t, cusolverDnSSgesv_bufferSize,
        cusolverDnHandle_t, handle,
        cusolver_int_t, n, cusolver_int_t, nrhs,
        float*, dA, cusolver_int_t, ldda,
        cusolver_int_t*, dipiv,
        float*, dB, cusolver_int_t, lddb,
        float*, dX, cusolver_int_t, lddx,
        void*, dWorkspace, size_t*, lwork_bytes)

DEF_FN(cusolverStatus_t, cusolverDnSHgesv_bufferSize,
        cusolverDnHandle_t, handle,
        cusolver_int_t, n, cusolver_int_t, nrhs,
        float*, dA, cusolver_int_t, ldda,
        cusolver_int_t*, dipiv,
        float*, dB, cusolver_int_t, lddb,
        float*, dX, cusolver_int_t, lddx,
        void*, dWorkspace, size_t*, lwork_bytes)

#if CUDART_VERSION >= 11000
DEF_FN(cusolverStatus_t, cusolverDnIRSXgesv,
        cusolverDnHandle_t, handle,
        cusolverDnIRSParams_t, gesv_irs_params,
        cusolverDnIRSInfos_t , gesv_irs_infos,
        int     n,
        int     nrhs,
        void   *dA,
        int     ldda,
        void   *dB,
        int     lddb,
        void   *dX,
        int     lddx,
        void   *dWorkspace,
        size_t  lwork_bytes,
        int    *dinfo)
#else
DEF_FN(cusolverStatus_t, cusolverDnIRSXgesv,
        cusolverDnHandle_t, handle,
        cusolverDnIRSParams_t, gesv_irs_params,
        cusolverDnIRSInfos_t , gesv_irs_infos,
        cudaDataType           , inout_data_type,
        cusolver_int_t, n, cusolver_int_t, nrhs,
        void*, dA, cusolver_int_t, ldda,
        cusolver_int_t*, dipiv,
        void*, dB, cusolver_int_t, lddb,
        void*, dX, cusolver_int_t, lddx,
        void*, dWorkspace, size_t, lwork_bytes,
        cusolver_int_t*, niters,
        cusolver_int_t*, d_info)
#endif

DEF_FN(cusolverStatus_t, cusolverDnIRSXgesv_bufferSize,
        cusolverDnHandle_t, handle,
        cusolverDnIRSParams_t, params,
        cusolver_int_t, n, cusolver_int_t, nrhs,
        size_t*, lwork_bytes)

DEF_FN(cusolverStatus_t, cusolverDnSpotrf_bufferSize, 
    cusolverDnHandle_t, handle, 
    cublasFillMode_t, uplo, 
    int, n, 
    float*, A, 
    int, lda, 
    int*, Lwork )

DEF_FN(cusolverStatus_t, cusolverDnDpotrf_bufferSize, 
    cusolverDnHandle_t, handle, 
    cublasFillMode_t, uplo, 
    int, n, 
    double*, A, 
    int, lda, 
    int*, Lwork )

DEF_FN(cusolverStatus_t, cusolverDnCpotrf_bufferSize, 
    cusolverDnHandle_t, handle, 
    cublasFillMode_t, uplo, 
    int, n, 
    cuComplex*, A, 
    int, lda, 
    int*, Lwork )

DEF_FN(cusolverStatus_t, cusolverDnZpotrf_bufferSize, 
    cusolverDnHandle_t, handle, 
    cublasFillMode_t, uplo, 
    int, n, 
    cuDoubleComplex*, A, 
    int, lda, 
    int*, Lwork)

DEF_FN(cusolverStatus_t, cusolverDnSpotrf, 
    cusolverDnHandle_t, handle, 
    cublasFillMode_t, uplo, 
    int, n, 
    float*, A, 
    int, lda,  
    float*, Workspace, 
    int, Lwork, 
    int*, devInfo )

DEF_FN(cusolverStatus_t, cusolverDnDpotrf, 
    cusolverDnHandle_t, handle, 
    cublasFillMode_t, uplo, 
    int, n, 
    double*, A, 
    int, lda, 
    double*, Workspace, 
    int, Lwork, 
    int*, devInfo )



DEF_FN(cusolverStatus_t, cusolverDnCpotrf, 
    cusolverDnHandle_t, handle, 
    cublasFillMode_t, uplo, 
    int, n, 
    cuComplex*, A, 
    int, lda, 
    cuComplex*, Workspace, 
    int, Lwork, 
    int*, devInfo )

DEF_FN(cusolverStatus_t, cusolverDnZpotrf, 
    cusolverDnHandle_t, handle, 
    cublasFillMode_t, uplo, 
    int, n, 
    cuDoubleComplex*, A, 
    int, lda, 
    cuDoubleComplex*, Workspace, 
    int, Lwork, 
    int*, devInfo )


DEF_FN(cusolverStatus_t, cusolverDnSpotrs,
    cusolverDnHandle_t, handle,
    cublasFillMode_t, uplo,
    int, n,
    int, nrhs,
    const float*, A,
    int, lda,
    float*, B,
    int, ldb,
    int*, devInfo)

DEF_FN(cusolverStatus_t, cusolverDnDpotrs,
    cusolverDnHandle_t, handle,
    cublasFillMode_t, uplo,
    int, n,
    int, nrhs,
    const double*, A,
    int, lda,
    double*, B,
    int, ldb,
    int*, devInfo)

DEF_FN(cusolverStatus_t, cusolverDnCpotrs,
    cusolverDnHandle_t, handle,
    cublasFillMode_t, uplo,
    int, n,
    int, nrhs,
    const cuComplex*, A,
    int, lda,
    cuComplex*, B,
    int, ldb,
    int*, devInfo)

DEF_FN(cusolverStatus_t, cusolverDnZpotrs,
    cusolverDnHandle_t, handle,
    cublasFillMode_t, uplo,
    int, n,
    int, nrhs,
    const cuDoubleComplex*, A,
    int, lda,
    cuDoubleComplex*, B,
    int, ldb,
    int*, devInfo)

DEF_FN(cusolverStatus_t, cusolverDnSpotrfBatched,
    cusolverDnHandle_t, handle,
    cublasFillMode_t, uplo,
    int, n,
    float* Aarray[],
    int, lda,
    int*, infoArray,
    int, batchSize)

DEF_FN(cusolverStatus_t, cusolverDnDpotrfBatched,
    cusolverDnHandle_t, handle,
    cublasFillMode_t, uplo,
    int, n,
    double* Aarray[],
    int, lda,
    int*, infoArray,
    int, batchSize)

DEF_FN(cusolverStatus_t, cusolverDnCpotrfBatched,
    cusolverDnHandle_t, handle,
    cublasFillMode_t, uplo,
    int, n,
    cuComplex* Aarray[],
    int, lda,
    int*, infoArray,
    int, batchSize)

DEF_FN(cusolverStatus_t, cusolverDnZpotrfBatched,
    cusolverDnHandle_t, handle,
    cublasFillMode_t, uplo,
    int, n,
    cuDoubleComplex**, Aarray,
    int, lda,
    int*, infoArray,
    int, batchSize)

DEF_FN(cusolverStatus_t, cusolverDnSpotrsBatched,
    cusolverDnHandle_t, handle,
    cublasFillMode_t, uplo,
    int, n,
    int, nrhs,     float**, A,
    int, lda,
    float**, B,
    int, ldb,
    int*, d_info,
    int, batchSize)

DEF_FN(cusolverStatus_t, cusolverDnDpotrsBatched,
    cusolverDnHandle_t, handle,
    cublasFillMode_t, uplo,
    int, n,
    int, nrhs,     double**, A,
    int, lda,
    double* B[],
    int, ldb,
    int*, d_info,
    int, batchSize)

DEF_FN(cusolverStatus_t, cusolverDnCpotrsBatched,
    cusolverDnHandle_t, handle,
    cublasFillMode_t, uplo,
    int, n,
    int, nrhs,     cuComplex**, A,
    int, lda,
    cuComplex* B[],
    int, ldb,
    int*, d_info,
    int, batchSize)

DEF_FN(cusolverStatus_t, cusolverDnZpotrsBatched,
    cusolverDnHandle_t, handle,
    cublasFillMode_t, uplo,
    int, n,
    int, nrhs,     cuDoubleComplex**, A,
    int, lda,
    cuDoubleComplex**, B,
    int, ldb,
    int*, d_info,
    int, batchSize)

DEF_FN(cusolverStatus_t, cusolverDnSpotri_bufferSize,
    cusolverDnHandle_t, handle,
    cublasFillMode_t, uplo,
    int, n,
    float*, A,
    int, lda,
    int*, lwork)

DEF_FN(cusolverStatus_t, cusolverDnDpotri_bufferSize,
    cusolverDnHandle_t, handle,
    cublasFillMode_t, uplo,
    int, n,
    double*, A,
    int, lda,
    int*, lwork)

DEF_FN(cusolverStatus_t, cusolverDnCpotri_bufferSize,
    cusolverDnHandle_t, handle,
    cublasFillMode_t, uplo,
    int, n,
    cuComplex*, A,
    int, lda,
    int*, lwork)

DEF_FN(cusolverStatus_t, cusolverDnZpotri_bufferSize,
    cusolverDnHandle_t, handle,
    cublasFillMode_t, uplo,
    int, n,
    cuDoubleComplex*, A,
    int, lda,
    int*, lwork)

DEF_FN(cusolverStatus_t, cusolverDnSpotri,
    cusolverDnHandle_t, handle,
    cublasFillMode_t, uplo,
    int, n,
    float*, A,
    int, lda,
    float*, work,
    int, lwork,
    int*, devInfo)

DEF_FN(cusolverStatus_t, cusolverDnDpotri,
    cusolverDnHandle_t, handle,
    cublasFillMode_t, uplo,
    int, n,
    double*, A,
    int, lda,
    double*, work,
    int, lwork,
    int*, devInfo)

DEF_FN(cusolverStatus_t, cusolverDnCpotri,
    cusolverDnHandle_t, handle,
    cublasFillMode_t, uplo,
    int, n,
    cuComplex*, A,
    int, lda,
    cuComplex*, work,
    int, lwork,
    int*, devInfo)

DEF_FN(cusolverStatus_t, cusolverDnZpotri,
    cusolverDnHandle_t, handle,
    cublasFillMode_t, uplo,
    int, n,
    cuDoubleComplex*, A,
    int, lda,
    cuDoubleComplex*, work,
    int, lwork,
    int*, devInfo)

DEF_FN(cusolverStatus_t, cusolverDnStrtri_bufferSize,
    cusolverDnHandle_t, handle,
    cublasFillMode_t, uplo,
    cublasDiagType_t, diag,
    int, n,
    float*, A,
    int, lda,
    int*, lwork)

DEF_FN(cusolverStatus_t, cusolverDnDtrtri_bufferSize,
    cusolverDnHandle_t, handle,
    cublasFillMode_t, uplo,
    cublasDiagType_t, diag,
    int, n,
    double*, A,
    int, lda,
    int*, lwork)

DEF_FN(cusolverStatus_t, cusolverDnCtrtri_bufferSize,
    cusolverDnHandle_t, handle,
    cublasFillMode_t, uplo,
    cublasDiagType_t, diag,
    int, n,
    cuComplex*, A,
    int, lda,
    int*, lwork)

DEF_FN(cusolverStatus_t, cusolverDnZtrtri_bufferSize,
    cusolverDnHandle_t, handle,
    cublasFillMode_t, uplo,
    cublasDiagType_t, diag,
    int, n,
    cuDoubleComplex*, A,
    int, lda,
    int*, lwork)

DEF_FN(cusolverStatus_t, cusolverDnStrtri,
    cusolverDnHandle_t, handle,
    cublasFillMode_t, uplo,
    cublasDiagType_t, diag,
    int, n,
    float*, A,
    int, lda,
    float*, work,
    int, lwork,
    int*, devInfo)

DEF_FN(cusolverStatus_t, cusolverDnDtrtri,
    cusolverDnHandle_t, handle,
    cublasFillMode_t, uplo,
    cublasDiagType_t, diag,
    int, n,
    double*, A,
    int, lda,
    double*, work,
    int, lwork,
    int*, devInfo)

DEF_FN(cusolverStatus_t, cusolverDnCtrtri,
    cusolverDnHandle_t, handle,
    cublasFillMode_t, uplo,
    cublasDiagType_t, diag,
    int, n,
    cuComplex*, A,
    int, lda,
    cuComplex*, work,
    int, lwork,
    int*, devInfo)

DEF_FN(cusolverStatus_t, cusolverDnZtrtri,
    cusolverDnHandle_t, handle,
    cublasFillMode_t, uplo,
    cublasDiagType_t, diag,
    int, n,
    cuDoubleComplex*, A,
    int, lda,
    cuDoubleComplex*, work,
    int, lwork,
    int*, devInfo)

DEF_FN(cusolverStatus_t, cusolverDnSlauum_bufferSize,
    cusolverDnHandle_t, handle,
    cublasFillMode_t, uplo,
    int, n,
    float*, A,
    int, lda,
    int*, lwork)

DEF_FN(cusolverStatus_t, cusolverDnDlauum_bufferSize,
    cusolverDnHandle_t, handle,
    cublasFillMode_t, uplo,
    int, n,
    double*, A,
    int, lda,
    int*, lwork)

DEF_FN(cusolverStatus_t, cusolverDnClauum_bufferSize,
    cusolverDnHandle_t, handle,
    cublasFillMode_t, uplo,
    int, n,
    cuComplex*, A,
    int, lda,
    int*, lwork)

DEF_FN(cusolverStatus_t, cusolverDnZlauum_bufferSize,
    cusolverDnHandle_t, handle,
    cublasFillMode_t, uplo,
    int, n,
    cuDoubleComplex*, A,
    int, lda,
    int*, lwork)

DEF_FN(cusolverStatus_t, cusolverDnSlauum,
    cusolverDnHandle_t, handle,
    cublasFillMode_t, uplo,
    int, n,
    float*, A,
    int, lda,
    float*, work,
    int, lwork,
    int*, devInfo)

DEF_FN(cusolverStatus_t, cusolverDnDlauum,
    cusolverDnHandle_t, handle,
    cublasFillMode_t, uplo,
    int, n,
    double*, A,
    int, lda,
    double*, work,
    int, lwork,
    int*, devInfo)

DEF_FN(cusolverStatus_t, cusolverDnClauum,
    cusolverDnHandle_t, handle,
    cublasFillMode_t, uplo,
    int, n,
    cuComplex*, A,
    int, lda,
    cuComplex*, work,
    int, lwork,
    int*, devInfo)

DEF_FN(cusolverStatus_t, cusolverDnZlauum,
    cusolverDnHandle_t, handle,
    cublasFillMode_t, uplo,
    int, n,
    cuDoubleComplex*, A,
    int, lda,
    cuDoubleComplex*, work,
    int, lwork,
    int*, devInfo)



DEF_FN(cusolverStatus_t, cusolverDnSgetrf_bufferSize,
    cusolverDnHandle_t, handle,
    int, m,
    int, n,
    float*, A,
    int, lda,
    int*, Lwork )

cusolverStatus_t cusolverDnDgetrf_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    double* A,
    int lda,
    int* Lwork )
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int_result result;
    enum clnt_stat retval_1;
    retval_1 = rpc_cusolverdndgetrf_buffersize_1((ptr)handle, m, n, (ptr)A, lda, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    if (result.err == 0) {
        *Lwork = result.int_result_u.data;
    }
    return result.err;
}

DEF_FN(cusolverStatus_t, cusolverDnCgetrf_bufferSize,
    cusolverDnHandle_t, handle,
    int, m,
    int, n,
    cuComplex*, A,
    int, lda,
    int*, Lwork )

DEF_FN(cusolverStatus_t, cusolverDnZgetrf_bufferSize,
    cusolverDnHandle_t, handle,
    int, m,
    int, n,
    cuDoubleComplex*, A,
    int, lda,
    int*, Lwork )


DEF_FN(cusolverStatus_t, cusolverDnSgetrf, 
    cusolverDnHandle_t, handle, 
    int, m, 
    int, n, 
    float*, A, 
    int, lda, 
    float*, Workspace, 
    int*, devIpiv, 
    int*, devInfo )

cusolverStatus_t cusolverDnDgetrf(
    cusolverDnHandle_t handle, 
    int m, 
    int n, 
    double* A, 
    int lda, 
    double* Workspace, 
    int* devIpiv, 
    int* devInfo )
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval_1;
    retval_1 = rpc_cusolverdndgetrf_1((ptr)handle, m, n, (ptr)A, lda, (ptr)Workspace, (ptr)devIpiv, (ptr)devInfo, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    return result;
}

DEF_FN(cusolverStatus_t, cusolverDnCgetrf, 
    cusolverDnHandle_t, handle, 
    int, m, 
    int, n, 
    cuComplex*, A, 
    int, lda, 
    cuComplex*, Workspace, 
    int*, devIpiv, 
    int*, devInfo )

DEF_FN(cusolverStatus_t, cusolverDnZgetrf, 
    cusolverDnHandle_t, handle, 
    int, m, 
    int, n, 
    cuDoubleComplex*, A, 
    int, lda, 
    cuDoubleComplex*, Workspace, 
    int*, devIpiv, 
    int*, devInfo )

DEF_FN(cusolverStatus_t, cusolverDnSlaswp, 
    cusolverDnHandle_t, handle, 
    int, n, 
    float*, A, 
    int, lda, 
    int, k1, 
    int, k2, 
    const int*, devIpiv, 
    int, incx)

DEF_FN(cusolverStatus_t, cusolverDnDlaswp, 
    cusolverDnHandle_t, handle, 
    int, n, 
    double*, A, 
    int, lda, 
    int, k1, 
    int, k2, 
    const int*, devIpiv, 
    int, incx)

DEF_FN(cusolverStatus_t, cusolverDnClaswp, 
    cusolverDnHandle_t, handle, 
    int, n, 
    cuComplex*, A, 
    int, lda, 
    int, k1, 
    int, k2, 
    const int*, devIpiv, 
    int, incx)

DEF_FN(cusolverStatus_t, cusolverDnZlaswp, 
    cusolverDnHandle_t, handle, 
    int, n, 
    cuDoubleComplex*, A, 
    int, lda, 
    int, k1, 
    int, k2, 
    const int*, devIpiv, 
    int, incx)

DEF_FN(cusolverStatus_t, cusolverDnSgetrs, 
    cusolverDnHandle_t, handle, 
    cublasOperation_t, trans, 
    int, n, 
    int, nrhs, 
    const float*, A, 
    int, lda, 
    const int*, devIpiv, 
    float*, B, 
    int, ldb, 
    int*, devInfo )

cusolverStatus_t cusolverDnDgetrs( 
    cusolverDnHandle_t handle, 
    cublasOperation_t trans, 
    int n, 
    int nrhs, 
    const double* A, 
    int lda, 
    const int* devIpiv, 
    double* B, 
    int ldb, 
    int* devInfo )
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval_1;
    retval_1 = rpc_cusolverdndgetrs_1((ptr)handle, (int)trans, n, nrhs, (ptr)A, lda, (ptr)devIpiv, (ptr)B, ldb, (ptr)devInfo, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    return result;
}

DEF_FN(cusolverStatus_t, cusolverDnCgetrs, 
    cusolverDnHandle_t, handle, 
    cublasOperation_t, trans, 
    int, n, 
    int, nrhs, 
    const cuComplex*, A, 
    int, lda, 
    const int*, devIpiv, 
    cuComplex*, B, 
    int, ldb, 
    int*, devInfo )

DEF_FN(cusolverStatus_t, cusolverDnZgetrs, 
    cusolverDnHandle_t, handle, 
    cublasOperation_t, trans, 
    int, n, 
    int, nrhs, 
    const cuDoubleComplex*, A, 
    int, lda, 
    const int*, devIpiv, 
    cuDoubleComplex*, B, 
    int, ldb, 
    int*, devInfo )


DEF_FN(cusolverStatus_t, cusolverDnSgeqrf_bufferSize,
    cusolverDnHandle_t, handle,
    int, m,
    int, n,
    float*, A,
    int, lda,
    int*, lwork )

DEF_FN(cusolverStatus_t, cusolverDnDgeqrf_bufferSize,
    cusolverDnHandle_t, handle,
    int, m,
    int, n,
    double*, A,
    int, lda,
    int*, lwork )

DEF_FN(cusolverStatus_t, cusolverDnCgeqrf_bufferSize,
    cusolverDnHandle_t, handle,
    int, m,
    int, n,
    cuComplex*, A,
    int, lda,
    int*, lwork )

DEF_FN(cusolverStatus_t, cusolverDnZgeqrf_bufferSize,
    cusolverDnHandle_t, handle,
    int, m,
    int, n,
    cuDoubleComplex*, A,
    int, lda,
    int*, lwork )

DEF_FN(cusolverStatus_t, cusolverDnSgeqrf, 
    cusolverDnHandle_t, handle, 
    int, m, 
    int, n, 
    float*, A,  
    int, lda, 
    float*, TAU,  
    float*, Workspace,  
    int, Lwork, 
    int*, devInfo )

DEF_FN(cusolverStatus_t, cusolverDnDgeqrf, 
    cusolverDnHandle_t, handle, 
    int, m, 
    int, n, 
    double*, A, 
    int, lda, 
    double*, TAU, 
    double*, Workspace, 
    int, Lwork, 
    int*, devInfo )

DEF_FN(cusolverStatus_t, cusolverDnCgeqrf, 
    cusolverDnHandle_t, handle, 
    int, m, 
    int, n, 
    cuComplex*, A, 
    int, lda, 
    cuComplex*, TAU, 
    cuComplex*, Workspace, 
    int, Lwork, 
    int*, devInfo )

DEF_FN(cusolverStatus_t, cusolverDnZgeqrf, 
    cusolverDnHandle_t, handle, 
    int, m, 
    int, n, 
    cuDoubleComplex*, A, 
    int, lda, 
    cuDoubleComplex*, TAU, 
    cuDoubleComplex*, Workspace, 
    int, Lwork, 
    int*, devInfo )


DEF_FN(cusolverStatus_t, cusolverDnSorgqr_bufferSize,
    cusolverDnHandle_t, handle,
    int, m,
    int, n,
    int, k,
    const float*, A,
    int, lda,
    const float*, tau,
    int*, lwork)

DEF_FN(cusolverStatus_t, cusolverDnDorgqr_bufferSize,
    cusolverDnHandle_t, handle,
    int, m,
    int, n,
    int, k,
    const double*, A,
    int, lda,
    const double*, tau,
    int*, lwork)

DEF_FN(cusolverStatus_t, cusolverDnCungqr_bufferSize,
    cusolverDnHandle_t, handle,
    int, m,
    int, n,
    int, k,
    const cuComplex*, A,
    int, lda,
    const cuComplex*, tau,
    int*, lwork)

DEF_FN(cusolverStatus_t, cusolverDnZungqr_bufferSize,
    cusolverDnHandle_t, handle,
    int, m,
    int, n,
    int, k,
    const cuDoubleComplex*, A,
    int, lda,
    const cuDoubleComplex*, tau,
    int*, lwork)

DEF_FN(cusolverStatus_t, cusolverDnSorgqr,
    cusolverDnHandle_t, handle,
    int, m,
    int, n,
    int, k,
    float*, A,
    int, lda,
    const float*, tau,
    float*, work,
    int, lwork,
    int*, info)

DEF_FN(cusolverStatus_t, cusolverDnDorgqr,
    cusolverDnHandle_t, handle,
    int, m,
    int, n,
    int, k,
    double*, A,
    int, lda,
    const double*, tau,
    double*, work,
    int, lwork,
    int*, info)

DEF_FN(cusolverStatus_t, cusolverDnCungqr,
    cusolverDnHandle_t, handle,
    int, m,
    int, n,
    int, k,
    cuComplex*, A,
    int, lda,
    const cuComplex*, tau,
    cuComplex*, work,
    int, lwork,
    int*, info)

DEF_FN(cusolverStatus_t, cusolverDnZungqr,
    cusolverDnHandle_t, handle,
    int, m,
    int, n,
    int, k,
    cuDoubleComplex*, A,
    int, lda,
    const cuDoubleComplex*, tau,
    cuDoubleComplex*, work,
    int, lwork,
    int*, info)



DEF_FN(cusolverStatus_t, cusolverDnSormqr_bufferSize,
    cusolverDnHandle_t, handle,
    cublasSideMode_t, side,
    cublasOperation_t, trans,
    int, m,
    int, n,
    int, k,
    const float*, A,
    int, lda,
    const float*, tau,
    const float*, C,
    int, ldc,
    int*, lwork)

DEF_FN(cusolverStatus_t, cusolverDnDormqr_bufferSize,
    cusolverDnHandle_t, handle,
    cublasSideMode_t, side,
    cublasOperation_t, trans,
    int, m,
    int, n,
    int, k,
    const double*, A,
    int, lda,
    const double*, tau,
    const double*, C,
    int, ldc,
    int*, lwork)

DEF_FN(cusolverStatus_t, cusolverDnCunmqr_bufferSize,
    cusolverDnHandle_t, handle,
    cublasSideMode_t, side,
    cublasOperation_t, trans,
    int, m,
    int, n,
    int, k,
    const cuComplex*, A,
    int, lda,
    const cuComplex*, tau,
    const cuComplex*, C,
    int, ldc,
    int*, lwork)

DEF_FN(cusolverStatus_t, cusolverDnZunmqr_bufferSize,
    cusolverDnHandle_t, handle,
    cublasSideMode_t, side,
    cublasOperation_t, trans,
    int, m,
    int, n,
    int, k,
    const cuDoubleComplex*, A,
    int, lda,
    const cuDoubleComplex*, tau,
    const cuDoubleComplex*, C,
    int, ldc,
    int*, lwork)

DEF_FN(cusolverStatus_t, cusolverDnSormqr,
    cusolverDnHandle_t, handle,
    cublasSideMode_t, side,
    cublasOperation_t, trans,
    int, m,
    int, n,
    int, k,
    const float*, A,
    int, lda,
    const float*, tau,
    float*, C,
    int, ldc,
    float*, work,
    int, lwork,
    int*, devInfo)

DEF_FN(cusolverStatus_t, cusolverDnDormqr,
    cusolverDnHandle_t, handle,
    cublasSideMode_t, side,
    cublasOperation_t, trans,
    int, m,
    int, n,
    int, k,
    const double*, A,
    int, lda,
    const double*, tau,
    double*, C,
    int, ldc,
    double*, work,
    int, lwork,
    int*, devInfo)

DEF_FN(cusolverStatus_t, cusolverDnCunmqr,
    cusolverDnHandle_t, handle,
    cublasSideMode_t, side,
    cublasOperation_t, trans,
    int, m,
    int, n,
    int, k,
    const cuComplex*, A,
    int, lda,
    const cuComplex*, tau,
    cuComplex*, C,
    int, ldc,
    cuComplex*, work,
    int, lwork,
    int*, devInfo)

DEF_FN(cusolverStatus_t, cusolverDnZunmqr,
    cusolverDnHandle_t, handle,
    cublasSideMode_t, side,
    cublasOperation_t, trans,
    int, m,
    int, n,
    int, k,
    const cuDoubleComplex*, A,
    int, lda,
    const cuDoubleComplex*, tau,
    cuDoubleComplex*, C,
    int, ldc,
    cuDoubleComplex*, work,
    int, lwork,
    int*, devInfo)


DEF_FN(cusolverStatus_t, cusolverDnSsytrf_bufferSize,
    cusolverDnHandle_t, handle,
    int, n,
    float*, A,
    int, lda,
    int*, lwork )

DEF_FN(cusolverStatus_t, cusolverDnDsytrf_bufferSize,
    cusolverDnHandle_t, handle,
    int, n,
    double*, A,
    int, lda,
    int*, lwork )

DEF_FN(cusolverStatus_t, cusolverDnCsytrf_bufferSize,
    cusolverDnHandle_t, handle,
    int, n,
    cuComplex*, A,
    int, lda,
    int*, lwork )

DEF_FN(cusolverStatus_t, cusolverDnZsytrf_bufferSize,
    cusolverDnHandle_t, handle,
    int, n,
    cuDoubleComplex*, A,
    int, lda,
    int*, lwork )

DEF_FN(cusolverStatus_t, cusolverDnSsytrf,
    cusolverDnHandle_t, handle,
    cublasFillMode_t, uplo,
    int, n,
    float*, A,
    int, lda,
    int*, ipiv,
    float*, work,
    int, lwork,
    int*, info )

DEF_FN(cusolverStatus_t, cusolverDnDsytrf,
    cusolverDnHandle_t, handle,
    cublasFillMode_t, uplo,
    int, n,
    double*, A,
    int, lda,
    int*, ipiv,
    double*, work,
    int, lwork,
    int*, info )

DEF_FN(cusolverStatus_t, cusolverDnCsytrf,
    cusolverDnHandle_t, handle,
    cublasFillMode_t, uplo,
    int, n,
    cuComplex*, A,
    int, lda,
    int*, ipiv,
    cuComplex*, work,
    int, lwork,
    int*, info )

DEF_FN(cusolverStatus_t, cusolverDnZsytrf,
    cusolverDnHandle_t, handle,
    cublasFillMode_t, uplo,
    int, n,
    cuDoubleComplex*, A,
    int, lda,
    int*, ipiv,
    cuDoubleComplex*, work,
    int, lwork,
    int*, info )

DEF_FN(cusolverStatus_t, cusolverDnSsytrs_bufferSize,
        cusolverDnHandle_t, handle,
        cublasFillMode_t, uplo,
        int, n,
        int, nrhs,
        const float*, A,
        int, lda,
        const int*, ipiv,
        float*, B,
        int, ldb,
        int*, lwork)

DEF_FN(cusolverStatus_t, cusolverDnDsytrs_bufferSize,
        cusolverDnHandle_t, handle,
        cublasFillMode_t, uplo,
        int, n,
        int, nrhs,
        const double*, A,
        int, lda,
        const int*, ipiv,
        double*, B,
        int, ldb,
        int*, lwork)

DEF_FN(cusolverStatus_t, cusolverDnCsytrs_bufferSize,
        cusolverDnHandle_t, handle,
        cublasFillMode_t, uplo,
        int, n,
        int, nrhs,
        const cuComplex*, A,
        int, lda,
        const int*, ipiv,
        cuComplex*, B,
        int, ldb,
        int*, lwork)

DEF_FN(cusolverStatus_t, cusolverDnZsytrs_bufferSize,
        cusolverDnHandle_t, handle,
        cublasFillMode_t, uplo,
        int, n,
        int, nrhs,
        const cuDoubleComplex*, A,
        int, lda,
        const int*, ipiv,
        cuDoubleComplex*, B,
        int, ldb,
        int*, lwork)

DEF_FN(cusolverStatus_t, cusolverDnSsytrs,
        cusolverDnHandle_t, handle,
        cublasFillMode_t, uplo,
        int, n,
        int, nrhs,
        const float*, A,
        int, lda,
        const int*, ipiv,
        float*, B,
        int, ldb,
        float*, work,
        int, lwork,
        int*, info)

DEF_FN(cusolverStatus_t, cusolverDnDsytrs,
        cusolverDnHandle_t, handle,
        cublasFillMode_t, uplo,
        int, n,
        int, nrhs,
        const double*, A,
        int, lda,
        const int*, ipiv,
        double*, B,
        int, ldb,
        double*, work,
        int, lwork,
        int*, info)

DEF_FN(cusolverStatus_t, cusolverDnCsytrs,
        cusolverDnHandle_t, handle,
        cublasFillMode_t, uplo,
        int, n,
        int, nrhs,
        const cuComplex*, A,
        int, lda,
        const int*, ipiv,
        cuComplex*, B,
        int, ldb,
        cuComplex*, work,
        int, lwork,
        int*, info)

DEF_FN(cusolverStatus_t, cusolverDnZsytrs,
        cusolverDnHandle_t, handle,
        cublasFillMode_t, uplo,
        int, n,
        int, nrhs,
        const cuDoubleComplex*, A,
        int, lda,
        const int*, ipiv,
        cuDoubleComplex*, B,
        int, ldb,
        cuDoubleComplex*, work,
        int, lwork,
        int*, info)

DEF_FN(cusolverStatus_t, cusolverDnSsytri_bufferSize,
        cusolverDnHandle_t, handle,
        cublasFillMode_t, uplo,
        int, n,
        float*, A,
        int, lda,
        const int*, ipiv,
        int*, lwork)

DEF_FN(cusolverStatus_t, cusolverDnDsytri_bufferSize,
        cusolverDnHandle_t, handle,
        cublasFillMode_t, uplo,
        int, n,
        double*, A,
        int, lda,
        const int*, ipiv,
        int*, lwork)

DEF_FN(cusolverStatus_t, cusolverDnCsytri_bufferSize,
        cusolverDnHandle_t, handle,
        cublasFillMode_t, uplo,
        int, n,
        cuComplex*, A,
        int, lda,
        const int*, ipiv,
        int*, lwork)

DEF_FN(cusolverStatus_t, cusolverDnZsytri_bufferSize,
        cusolverDnHandle_t, handle,
        cublasFillMode_t, uplo,
        int, n,
        cuDoubleComplex*, A,
        int, lda,
        const int*, ipiv,
        int*, lwork)

DEF_FN(cusolverStatus_t, cusolverDnSsytri,
        cusolverDnHandle_t, handle,
        cublasFillMode_t, uplo,
        int, n,
        float*, A,
        int, lda,
        const int*, ipiv,
        float*, work,
        int, lwork,
        int*, info)

DEF_FN(cusolverStatus_t, cusolverDnDsytri,
        cusolverDnHandle_t, handle,
        cublasFillMode_t, uplo,
        int, n,
        double*, A,
        int, lda,
        const int*, ipiv,
        double*, work,
        int, lwork,
        int*, info)

DEF_FN(cusolverStatus_t, cusolverDnCsytri,
        cusolverDnHandle_t, handle,
        cublasFillMode_t, uplo,
        int, n,
        cuComplex*, A,
        int, lda,
        const int*, ipiv,
        cuComplex*, work,
        int, lwork,
        int*, info)

DEF_FN(cusolverStatus_t, cusolverDnZsytri,
        cusolverDnHandle_t, handle,
        cublasFillMode_t, uplo,
        int, n,
        cuDoubleComplex*, A,
        int, lda,
        const int*, ipiv,
        cuDoubleComplex*, work,
        int, lwork,
        int*, info)


DEF_FN(cusolverStatus_t, cusolverDnSgebrd_bufferSize,
    cusolverDnHandle_t, handle,
    int, m,
    int, n,
    int*, Lwork )

DEF_FN(cusolverStatus_t, cusolverDnDgebrd_bufferSize,
    cusolverDnHandle_t, handle,
    int, m,
    int, n,
    int*, Lwork )

DEF_FN(cusolverStatus_t, cusolverDnCgebrd_bufferSize,
    cusolverDnHandle_t, handle,
    int, m,
    int, n,
    int*, Lwork )

DEF_FN(cusolverStatus_t, cusolverDnZgebrd_bufferSize,
    cusolverDnHandle_t, handle,
    int, m,
    int, n,
    int*, Lwork )

DEF_FN(cusolverStatus_t, cusolverDnSgebrd, 
    cusolverDnHandle_t, handle, 
    int, m, 
    int, n, 
    float*, A,  
    int, lda,
    float*, D, 
    float*, E, 
    float*, TAUQ,  
    float*, TAUP, 
    float*, Work,
    int, Lwork, 
    int*, devInfo )

DEF_FN(cusolverStatus_t, cusolverDnDgebrd, 
    cusolverDnHandle_t, handle, 
    int, m, 
    int, n, 
    double*, A, 
    int, lda,
    double*, D, 
    double*, E, 
    double*, TAUQ, 
    double*, TAUP, 
    double*, Work,
    int, Lwork, 
    int*, devInfo )

DEF_FN(cusolverStatus_t, cusolverDnCgebrd, 
    cusolverDnHandle_t, handle, 
    int, m, 
    int, n, 
    cuComplex*, A, 
    int, lda, 
    float*, D, 
    float*, E, 
    cuComplex*, TAUQ, 
    cuComplex*, TAUP,
    cuComplex*, Work, 
    int, Lwork, 
    int*, devInfo )

DEF_FN(cusolverStatus_t, cusolverDnZgebrd, 
    cusolverDnHandle_t, handle, 
    int, m, 
    int, n, 
    cuDoubleComplex*, A,
    int, lda, 
    double*, D, 
    double*, E, 
    cuDoubleComplex*, TAUQ,
    cuDoubleComplex*, TAUP, 
    cuDoubleComplex*, Work, 
    int, Lwork, 
    int*, devInfo )

DEF_FN(cusolverStatus_t, cusolverDnSorgbr_bufferSize,
    cusolverDnHandle_t, handle,
    cublasSideMode_t, side, 
    int, m,
    int, n,
    int, k,
    const float*, A,
    int, lda,
    const float*, tau,
    int*, lwork)

DEF_FN(cusolverStatus_t, cusolverDnDorgbr_bufferSize,
    cusolverDnHandle_t, handle,
    cublasSideMode_t, side, 
    int, m,
    int, n,
    int, k,
    const double*, A,
    int, lda,
    const double*, tau,
    int*, lwork)

DEF_FN(cusolverStatus_t, cusolverDnCungbr_bufferSize,
    cusolverDnHandle_t, handle,
    cublasSideMode_t, side, 
    int, m,
    int, n,
    int, k,
    const cuComplex*, A,
    int, lda,
    const cuComplex*, tau,
    int*, lwork)

DEF_FN(cusolverStatus_t, cusolverDnZungbr_bufferSize,
    cusolverDnHandle_t, handle,
    cublasSideMode_t, side, 
    int, m,
    int, n,
    int, k,
    const cuDoubleComplex*, A,
    int, lda,
    const cuDoubleComplex*, tau,
    int*, lwork)

DEF_FN(cusolverStatus_t, cusolverDnSorgbr,
    cusolverDnHandle_t, handle,
    cublasSideMode_t, side, 
    int, m,
    int, n,
    int, k,
    float*, A,
    int, lda,
    const float*, tau,
    float*, work,
    int, lwork,
    int*, info)

DEF_FN(cusolverStatus_t, cusolverDnDorgbr,
    cusolverDnHandle_t, handle,
    cublasSideMode_t, side, 
    int, m,
    int, n,
    int, k,
    double*, A,
    int, lda,
    const double*, tau,
    double*, work,
    int, lwork,
    int*, info)

DEF_FN(cusolverStatus_t, cusolverDnCungbr,
    cusolverDnHandle_t, handle,
    cublasSideMode_t, side, 
    int, m,
    int, n,
    int, k,
    cuComplex*, A,
    int, lda,
    const cuComplex*, tau,
    cuComplex*, work,
    int, lwork,
    int*, info)

DEF_FN(cusolverStatus_t, cusolverDnZungbr,
    cusolverDnHandle_t, handle,
    cublasSideMode_t, side, 
    int, m,
    int, n,
    int, k,
    cuDoubleComplex*, A,
    int, lda,
    const cuDoubleComplex*, tau,
    cuDoubleComplex*, work,
    int, lwork,
    int*, info)


DEF_FN(cusolverStatus_t, cusolverDnSsytrd_bufferSize,
    cusolverDnHandle_t, handle,
    cublasFillMode_t, uplo,
    int, n,
    const float*, A,
    int, lda,
    const float*, d,
    const float*, e,
    const float*, tau,
    int*, lwork)

DEF_FN(cusolverStatus_t, cusolverDnDsytrd_bufferSize,
    cusolverDnHandle_t, handle,
    cublasFillMode_t, uplo,
    int, n,
    const double*, A,
    int, lda,
    const double*, d,
    const double*, e,
    const double*, tau,
    int*, lwork)

DEF_FN(cusolverStatus_t, cusolverDnChetrd_bufferSize,
    cusolverDnHandle_t, handle,
    cublasFillMode_t, uplo,
    int, n,
    const cuComplex*, A,
    int, lda,
    const float*, d,
    const float*, e,
    const cuComplex*, tau,
    int*, lwork)

DEF_FN(cusolverStatus_t, cusolverDnZhetrd_bufferSize,
    cusolverDnHandle_t, handle,
    cublasFillMode_t, uplo,
    int, n,
    const cuDoubleComplex*, A,
    int, lda,
    const double*, d,
    const double*, e,
    const cuDoubleComplex*, tau,
    int*, lwork)


DEF_FN(cusolverStatus_t, cusolverDnSsytrd,
    cusolverDnHandle_t, handle,
    cublasFillMode_t, uplo,
    int, n,
    float*, A,
    int, lda,
    float*, d,
    float*, e,
    float*, tau,
    float*, work,
    int, lwork,
    int*, info)

DEF_FN(cusolverStatus_t, cusolverDnDsytrd,
    cusolverDnHandle_t, handle,
    cublasFillMode_t, uplo,
    int, n,
    double*, A,
    int, lda,
    double*, d,
    double*, e,
    double*, tau,
    double*, work,
    int, lwork,
    int*, info)

DEF_FN(cusolverStatus_t, cusolverDnChetrd,
    cusolverDnHandle_t, handle,
    cublasFillMode_t, uplo,
    int, n,
    cuComplex*, A,
    int, lda,
    float*, d,
    float*, e,
    cuComplex*, tau,
    cuComplex*, work,
    int, lwork,
    int*, info)

DEF_FN(cusolverStatus_t, cusolverDnZhetrd,
    cusolverDnHandle_t, handle,
    cublasFillMode_t, uplo,
    int, n,
    cuDoubleComplex*, A,
    int, lda,
    double*, d,
    double*, e,
    cuDoubleComplex*, tau,
    cuDoubleComplex*, work,
    int, lwork,
    int*, info)



DEF_FN(cusolverStatus_t, cusolverDnSorgtr_bufferSize,
    cusolverDnHandle_t, handle,
    cublasFillMode_t, uplo, 
    int, n,
    const float*, A,
    int, lda,
    const float*, tau,
    int*, lwork)

DEF_FN(cusolverStatus_t, cusolverDnDorgtr_bufferSize,
    cusolverDnHandle_t, handle,
    cublasFillMode_t, uplo,
    int, n,
    const double*, A,
    int, lda,
    const double*, tau,
    int*, lwork)

DEF_FN(cusolverStatus_t, cusolverDnCungtr_bufferSize,
    cusolverDnHandle_t, handle,
    cublasFillMode_t, uplo, 
    int, n,
    const cuComplex*, A,
    int, lda,
    const cuComplex*, tau,
    int*, lwork)

DEF_FN(cusolverStatus_t, cusolverDnZungtr_bufferSize,
    cusolverDnHandle_t, handle,
    cublasFillMode_t, uplo, 
    int, n,
    const cuDoubleComplex*, A,
    int, lda,
    const cuDoubleComplex*, tau,
    int*, lwork)

DEF_FN(cusolverStatus_t, cusolverDnSorgtr,
    cusolverDnHandle_t, handle,
    cublasFillMode_t, uplo, 
    int, n,
    float*, A,
    int, lda,
    const float*, tau,
    float*, work,
    int, lwork,
    int*, info)

DEF_FN(cusolverStatus_t, cusolverDnDorgtr,
    cusolverDnHandle_t, handle,
    cublasFillMode_t, uplo, 
    int, n,
    double*, A,
    int, lda,
    const double*, tau,
    double*, work,
    int, lwork,
    int*, info)

DEF_FN(cusolverStatus_t, cusolverDnCungtr,
    cusolverDnHandle_t, handle,
    cublasFillMode_t, uplo, 
    int, n,
    cuComplex*, A,
    int, lda,
    const cuComplex*, tau,
    cuComplex*, work,
    int, lwork,
    int*, info)

DEF_FN(cusolverStatus_t, cusolverDnZungtr,
    cusolverDnHandle_t, handle,
    cublasFillMode_t, uplo, 
    int, n,
    cuDoubleComplex*, A,
    int, lda,
    const cuDoubleComplex*, tau,
    cuDoubleComplex*, work,
    int, lwork,
    int*, info)



DEF_FN(cusolverStatus_t, cusolverDnSormtr_bufferSize,
    cusolverDnHandle_t, handle,
    cublasSideMode_t, side,
    cublasFillMode_t, uplo,
    cublasOperation_t, trans,
    int, m,
    int, n,
    const float*, A,
    int, lda,
    const float*, tau,
    const float*, C,
    int, ldc,
    int*, lwork)

DEF_FN(cusolverStatus_t, cusolverDnDormtr_bufferSize,
    cusolverDnHandle_t, handle,
    cublasSideMode_t, side,
    cublasFillMode_t, uplo,
    cublasOperation_t, trans,
    int, m,
    int, n,
    const double*, A,
    int, lda,
    const double*, tau,
    const double*, C,
    int, ldc,
    int*, lwork)

DEF_FN(cusolverStatus_t, cusolverDnCunmtr_bufferSize,
    cusolverDnHandle_t, handle,
    cublasSideMode_t, side,
    cublasFillMode_t, uplo,
    cublasOperation_t, trans,
    int, m,
    int, n,
    const cuComplex*, A,
    int, lda,
    const cuComplex*, tau,
    const cuComplex*, C,
    int, ldc,
    int*, lwork)

DEF_FN(cusolverStatus_t, cusolverDnZunmtr_bufferSize,
    cusolverDnHandle_t, handle,
    cublasSideMode_t, side,
    cublasFillMode_t, uplo,
    cublasOperation_t, trans,
    int, m,
    int, n,
    const cuDoubleComplex*, A,
    int, lda,
    const cuDoubleComplex*, tau,
    const cuDoubleComplex*, C,
    int, ldc,
    int*, lwork)

DEF_FN(cusolverStatus_t, cusolverDnSormtr,
    cusolverDnHandle_t, handle,
    cublasSideMode_t, side,
    cublasFillMode_t, uplo,
    cublasOperation_t, trans,
    int, m,
    int, n,
    float*, A,
    int, lda,
    float*, tau,
    float*, C,
    int, ldc,
    float*, work,
    int, lwork,
    int*, info)

DEF_FN(cusolverStatus_t, cusolverDnDormtr,
    cusolverDnHandle_t, handle,
    cublasSideMode_t, side,
    cublasFillMode_t, uplo,
    cublasOperation_t, trans,
    int, m,
    int, n,
    double*, A,
    int, lda,
    double*, tau,
    double*, C,
    int, ldc,
    double*, work,
    int, lwork,
    int*, info)

DEF_FN(cusolverStatus_t, cusolverDnCunmtr,
    cusolverDnHandle_t, handle,
    cublasSideMode_t, side,
    cublasFillMode_t, uplo,
    cublasOperation_t, trans,
    int, m,
    int, n,
    cuComplex*, A,
    int, lda,
    cuComplex*, tau,
    cuComplex*, C,
    int, ldc,
    cuComplex*, work,
    int, lwork,
    int*, info)

DEF_FN(cusolverStatus_t, cusolverDnZunmtr,
    cusolverDnHandle_t, handle,
    cublasSideMode_t, side,
    cublasFillMode_t, uplo,
    cublasOperation_t, trans,
    int, m,
    int, n,
    cuDoubleComplex*, A,
    int, lda,
    cuDoubleComplex*, tau,
    cuDoubleComplex*, C,
    int, ldc,
    cuDoubleComplex*, work,
    int, lwork,
    int*, info)



DEF_FN(cusolverStatus_t, cusolverDnSgesvd_bufferSize,
    cusolverDnHandle_t, handle,
    int, m,
    int, n,
    int*, lwork )

DEF_FN(cusolverStatus_t, cusolverDnDgesvd_bufferSize,
    cusolverDnHandle_t, handle,
    int, m,
    int, n,
    int*, lwork )

DEF_FN(cusolverStatus_t, cusolverDnCgesvd_bufferSize,
    cusolverDnHandle_t, handle,
    int, m,
    int, n,
    int*, lwork )

DEF_FN(cusolverStatus_t, cusolverDnZgesvd_bufferSize,
    cusolverDnHandle_t, handle,
    int, m,
    int, n,
    int*, lwork )

DEF_FN(cusolverStatus_t, cusolverDnSgesvd ,
    cusolverDnHandle_t, handle, 
    signed char, jobu, 
    signed char, jobvt, 
    int, m, 
    int, n, 
    float*, A, 
    int, lda, 
    float*, S, 
    float*, U, 
    int, ldu, 
    float*, VT, 
    int, ldvt, 
    float*, work, 
    int, lwork, 
    float*, rwork, 
    int* , info )

DEF_FN(cusolverStatus_t, cusolverDnDgesvd ,
    cusolverDnHandle_t, handle, 
    signed char, jobu, 
    signed char, jobvt, 
    int, m, 
    int, n, 
    double*, A, 
    int, lda, 
    double*, S, 
    double*, U, 
    int, ldu, 
    double*, VT, 
    int, ldvt, 
    double*, work,
    int, lwork, 
    double*, rwork, 
    int*, info )

DEF_FN(cusolverStatus_t, cusolverDnCgesvd ,
    cusolverDnHandle_t, handle, 
    signed char, jobu, 
    signed char, jobvt, 
    int, m, 
    int, n, 
    cuComplex*, A,
    int, lda, 
    float*, S, 
    cuComplex*, U, 
    int, ldu, 
    cuComplex*, VT, 
    int, ldvt,
    cuComplex*, work, 
    int, lwork, 
    float*, rwork, 
    int*, info )

DEF_FN(cusolverStatus_t, cusolverDnZgesvd ,
    cusolverDnHandle_t, handle, 
    signed char, jobu, 
    signed char, jobvt, 
    int, m, 
    int, n, 
    cuDoubleComplex*, A, 
    int, lda, 
    double*, S, 
    cuDoubleComplex*, U, 
    int, ldu, 
    cuDoubleComplex*, VT, 
    int, ldvt, 
    cuDoubleComplex*, work, 
    int, lwork, 
    double*, rwork, 
    int*, info )


DEF_FN(cusolverStatus_t, cusolverDnSsyevd_bufferSize,
    cusolverDnHandle_t, handle,
    cusolverEigMode_t, jobz, 
    cublasFillMode_t, uplo, 
    int, n,
    const float*, A,
    int, lda,
    const float*, W,
    int*, lwork)

DEF_FN(cusolverStatus_t, cusolverDnDsyevd_bufferSize,
    cusolverDnHandle_t, handle,
    cusolverEigMode_t, jobz, 
    cublasFillMode_t, uplo, 
    int, n,
    const double*, A,
    int, lda,
    const double*, W,
    int*, lwork)

DEF_FN(cusolverStatus_t, cusolverDnCheevd_bufferSize,
    cusolverDnHandle_t, handle,
    cusolverEigMode_t, jobz, 
    cublasFillMode_t, uplo, 
    int, n,
    const cuComplex*, A,
    int, lda,
    const float*, W,
    int*, lwork)

DEF_FN(cusolverStatus_t, cusolverDnZheevd_bufferSize,
    cusolverDnHandle_t, handle,
    cusolverEigMode_t, jobz, 
    cublasFillMode_t, uplo, 
    int, n,
    const cuDoubleComplex*, A,
    int, lda,
    const double*, W,
    int*, lwork)

DEF_FN(cusolverStatus_t, cusolverDnSsyevd,
    cusolverDnHandle_t, handle,
    cusolverEigMode_t, jobz, 
    cublasFillMode_t, uplo, 
    int, n,
    float*, A,
    int, lda,
    float*, W, 
    float*, work,
    int, lwork,
    int*, info)

DEF_FN(cusolverStatus_t, cusolverDnDsyevd,
    cusolverDnHandle_t, handle,
    cusolverEigMode_t, jobz, 
    cublasFillMode_t, uplo, 
    int, n,
    double*, A,
    int, lda,
    double*, W, 
    double*, work,
    int, lwork,
    int*, info)

DEF_FN(cusolverStatus_t, cusolverDnCheevd,
    cusolverDnHandle_t, handle,
    cusolverEigMode_t, jobz, 
    cublasFillMode_t, uplo, 
    int, n,
    cuComplex*, A,
    int, lda,
    float*, W, 
    cuComplex*, work,
    int, lwork,
    int*, info)

DEF_FN(cusolverStatus_t, cusolverDnZheevd,
    cusolverDnHandle_t, handle,
    cusolverEigMode_t, jobz, 
    cublasFillMode_t, uplo, 
    int, n,
    cuDoubleComplex*, A,
    int, lda,
    double*, W, 
    cuDoubleComplex*, work,
    int, lwork,
    int*, info)

DEF_FN(cusolverStatus_t, cusolverDnSsyevdx_bufferSize,
    cusolverDnHandle_t, handle,
    cusolverEigMode_t, jobz, 
    cusolverEigRange_t, range,
    cublasFillMode_t, uplo, 
    int, n,
    const float*, A,
    int, lda,
    float, vl,
    float, vu,
    int, il,
    int, iu,
    int*, meig,
    const float*, W,
    int*, lwork)

DEF_FN(cusolverStatus_t, cusolverDnDsyevdx_bufferSize,
    cusolverDnHandle_t, handle,
    cusolverEigMode_t, jobz, 
    cusolverEigRange_t, range,
    cublasFillMode_t, uplo, 
    int, n,
    const double*, A,
    int, lda,
    double, vl,
    double, vu,
    int, il,
    int, iu,
    int*, meig,
    const double*, W,
    int*, lwork)

DEF_FN(cusolverStatus_t, cusolverDnCheevdx_bufferSize,
    cusolverDnHandle_t, handle,
    cusolverEigMode_t, jobz, 
    cusolverEigRange_t, range,
    cublasFillMode_t, uplo, 
    int, n,
    const cuComplex*, A,
    int, lda,
    float, vl,
    float, vu,
    int, il,
    int, iu,
    int*, meig,
    const float*, W,
    int*, lwork)

DEF_FN(cusolverStatus_t, cusolverDnZheevdx_bufferSize,
    cusolverDnHandle_t, handle,
    cusolverEigMode_t, jobz, 
    cusolverEigRange_t, range,
    cublasFillMode_t, uplo, 
    int, n,
    const cuDoubleComplex*, A,
    int, lda,
    double, vl,
    double, vu,
    int, il,
    int, iu,
    int*, meig,
    const double*, W,
    int*, lwork)

DEF_FN(cusolverStatus_t, cusolverDnSsyevdx,
    cusolverDnHandle_t, handle,
    cusolverEigMode_t, jobz, 
    cusolverEigRange_t, range,
    cublasFillMode_t, uplo, 
    int, n,
    float*, A,
    int, lda,
    float, vl,
    float, vu,
    int, il,
    int, iu,
    int*, meig,
    float*, W, 
    float*, work,
    int, lwork,
    int*, info)

DEF_FN(cusolverStatus_t, cusolverDnDsyevdx,
    cusolverDnHandle_t, handle,
    cusolverEigMode_t, jobz, 
    cusolverEigRange_t, range,
    cublasFillMode_t, uplo, 
    int, n,
    double*, A,
    int, lda,
    double, vl,
    double, vu,
    int, il,
    int, iu,
    int*, meig,
    double*, W, 
    double*, work,
    int, lwork,
    int*, info)

DEF_FN(cusolverStatus_t, cusolverDnCheevdx,
    cusolverDnHandle_t, handle,
    cusolverEigMode_t, jobz, 
    cusolverEigRange_t, range,
    cublasFillMode_t, uplo, 
    int, n,
    cuComplex*, A,
    int, lda,
    float, vl,
    float, vu,
    int, il,
    int, iu,
    int*, meig,
    float*, W, 
    cuComplex*, work,
    int, lwork,
    int*, info)

DEF_FN(cusolverStatus_t, cusolverDnZheevdx,
    cusolverDnHandle_t, handle,
    cusolverEigMode_t, jobz, 
    cusolverEigRange_t, range,
    cublasFillMode_t, uplo, 
    int, n,
    cuDoubleComplex*, A,
    int, lda,
    double, vl,
    double, vu,
    int, il,
    int, iu,
    int*, meig,
    double*, W, 
    cuDoubleComplex*, work,
    int, lwork,
    int*, info)

DEF_FN(cusolverStatus_t, cusolverDnSsygvdx_bufferSize,
    cusolverDnHandle_t, handle,
    cusolverEigType_t, itype,
    cusolverEigMode_t, jobz,
    cusolverEigRange_t, range,
    cublasFillMode_t, uplo, 
    int, n,
    const float*, A, 
    int, lda,
    const float*, B, 
    int, ldb,
    float, vl,
    float, vu,
    int, il,
    int, iu,
    int*, meig,
    const float*, W,
    int*, lwork)

DEF_FN(cusolverStatus_t, cusolverDnDsygvdx_bufferSize,
    cusolverDnHandle_t, handle,
    cusolverEigType_t, itype, 
    cusolverEigMode_t, jobz,  
    cusolverEigRange_t, range,
    cublasFillMode_t, uplo,  
    int, n,
    const double*, A, 
    int, lda,
    const double*, B, 
    int, ldb,
    double, vl,
    double, vu,
    int, il,
    int, iu,
    int*, meig,
    const double*, W,
    int*, lwork)

DEF_FN(cusolverStatus_t, cusolverDnChegvdx_bufferSize,
    cusolverDnHandle_t, handle,
    cusolverEigType_t, itype, 
    cusolverEigMode_t, jobz,  
    cusolverEigRange_t, range,
    cublasFillMode_t, uplo,  
    int, n,
    const cuComplex*, A, 
    int, lda,
    const cuComplex*, B, 
    int, ldb,
    float, vl,
    float, vu,
    int, il,
    int, iu,
    int*, meig,
    const float*, W,
    int*, lwork)

DEF_FN(cusolverStatus_t, cusolverDnZhegvdx_bufferSize,
    cusolverDnHandle_t, handle,
    cusolverEigType_t, itype,   
    cusolverEigMode_t, jobz, 
    cusolverEigRange_t, range,
    cublasFillMode_t, uplo,  
    int, n,
    const cuDoubleComplex*, A,
    int, lda,
    const cuDoubleComplex*, B, 
    int, ldb,
    double, vl,
    double, vu,
    int, il,
    int, iu,
    int*, meig,
    const double*, W,
    int*, lwork)


DEF_FN(cusolverStatus_t, cusolverDnSsygvdx,
    cusolverDnHandle_t, handle,
    cusolverEigType_t, itype,   
    cusolverEigMode_t, jobz,  
    cusolverEigRange_t, range,
    cublasFillMode_t, uplo,  
    int, n,
    float*, A, 
    int, lda,
    float*, B, 
    int, ldb,
    float, vl,
    float, vu,
    int, il,
    int, iu,
    int*, meig,
    float*, W, 
    float*, work,
    int, lwork,
    int*, info)

DEF_FN(cusolverStatus_t, cusolverDnDsygvdx,
    cusolverDnHandle_t, handle,
    cusolverEigType_t, itype,  
    cusolverEigMode_t, jobz,  
    cusolverEigRange_t, range,
    cublasFillMode_t, uplo,  
    int, n,
    double*, A, 
    int, lda,
    double*, B, 
    int, ldb,
    double, vl,
    double, vu,
    int, il,
    int, iu,
    int*, meig,
    double*, W, 
    double*, work,
    int, lwork,
    int*, info)

DEF_FN(cusolverStatus_t, cusolverDnChegvdx,
    cusolverDnHandle_t, handle,
    cusolverEigType_t, itype,   
    cusolverEigMode_t, jobz,  
    cusolverEigRange_t, range,
    cublasFillMode_t, uplo,  
    int, n,
    cuComplex*, A,
    int, lda,
    cuComplex*, B, 
    int, ldb,
    float, vl,
    float, vu,
    int, il,
    int, iu,
    int*, meig,
    float*, W, 
    cuComplex*, work,
    int, lwork,
    int*, info)

DEF_FN(cusolverStatus_t, cusolverDnZhegvdx,
    cusolverDnHandle_t, handle,
    cusolverEigType_t, itype,   
    cusolverEigMode_t, jobz,  
    cusolverEigRange_t, range,
    cublasFillMode_t, uplo,  
    int, n,
    cuDoubleComplex*, A, 
    int, lda,
    cuDoubleComplex*, B, 
    int, ldb,
    double, vl,
    double, vu,
    int, il,
    int, iu,
    int*, meig,
    double*, W, 
    cuDoubleComplex*, work,
    int, lwork,
    int*, info)


DEF_FN(cusolverStatus_t, cusolverDnSsygvd_bufferSize,
    cusolverDnHandle_t, handle,
    cusolverEigType_t, itype,
    cusolverEigMode_t, jobz,
    cublasFillMode_t, uplo, 
    int, n,
    const float*, A, 
    int, lda,
    const float*, B, 
    int, ldb,
    const float*, W,
    int*, lwork)

DEF_FN(cusolverStatus_t, cusolverDnDsygvd_bufferSize,
    cusolverDnHandle_t, handle,
    cusolverEigType_t, itype, 
    cusolverEigMode_t, jobz,  
    cublasFillMode_t, uplo,  
    int, n,
    const double*, A, 
    int, lda,
    const double*, B, 
    int, ldb,
    const double*, W,
    int*, lwork)

DEF_FN(cusolverStatus_t, cusolverDnChegvd_bufferSize,
    cusolverDnHandle_t, handle,
    cusolverEigType_t, itype, 
    cusolverEigMode_t, jobz,  
    cublasFillMode_t, uplo,  
    int, n,
    const cuComplex*, A, 
    int, lda,
    const cuComplex*, B, 
    int, ldb,
    const float*, W,
    int*, lwork)

DEF_FN(cusolverStatus_t, cusolverDnZhegvd_bufferSize,
    cusolverDnHandle_t, handle,
    cusolverEigType_t, itype,   
    cusolverEigMode_t, jobz, 
    cublasFillMode_t, uplo,  
    int, n,
    const cuDoubleComplex*, A,
    int, lda,
    const cuDoubleComplex*, B, 
    int, ldb,
    const double*, W,
    int*, lwork)


DEF_FN(cusolverStatus_t, cusolverDnSsygvd,
    cusolverDnHandle_t, handle,
    cusolverEigType_t, itype,   
    cusolverEigMode_t, jobz,  
    cublasFillMode_t, uplo,  
    int, n,
    float*, A, 
    int, lda,
    float*, B, 
    int, ldb,
    float*, W, 
    float*, work,
    int, lwork,
    int*, info)

DEF_FN(cusolverStatus_t, cusolverDnDsygvd,
    cusolverDnHandle_t, handle,
    cusolverEigType_t, itype,  
    cusolverEigMode_t, jobz,  
    cublasFillMode_t, uplo,  
    int, n,
    double*, A, 
    int, lda,
    double*, B, 
    int, ldb,
    double*, W, 
    double*, work,
    int, lwork,
    int*, info)

DEF_FN(cusolverStatus_t, cusolverDnChegvd,
    cusolverDnHandle_t, handle,
    cusolverEigType_t, itype,   
    cusolverEigMode_t, jobz,  
    cublasFillMode_t, uplo,  
    int, n,
    cuComplex*, A,
    int, lda,
    cuComplex*, B, 
    int, ldb,
    float*, W, 
    cuComplex*, work,
    int, lwork,
    int*, info)

DEF_FN(cusolverStatus_t, cusolverDnZhegvd,
    cusolverDnHandle_t, handle,
    cusolverEigType_t, itype,   
    cusolverEigMode_t, jobz,  
    cublasFillMode_t, uplo,  
    int, n,
    cuDoubleComplex*, A, 
    int, lda,
    cuDoubleComplex*, B, 
    int, ldb,
    double*, W, 
    cuDoubleComplex*, work,
    int, lwork,
    int*, info)


DEF_FN(cusolverStatus_t, cusolverDnCreateSyevjInfo,
    syevjInfo_t*, info)

DEF_FN(cusolverStatus_t, cusolverDnDestroySyevjInfo,
    syevjInfo_t, info)

DEF_FN(cusolverStatus_t, cusolverDnXsyevjSetTolerance,
    syevjInfo_t, info,
    double, tolerance)

DEF_FN(cusolverStatus_t, cusolverDnXsyevjSetMaxSweeps,
    syevjInfo_t, info,
    int, max_sweeps)

DEF_FN(cusolverStatus_t, cusolverDnXsyevjSetSortEig,
    syevjInfo_t, info,
    int, sort_eig)

DEF_FN(cusolverStatus_t, cusolverDnXsyevjGetResidual,
    cusolverDnHandle_t, handle,
    syevjInfo_t, info,
    double*, residual)

DEF_FN(cusolverStatus_t, cusolverDnXsyevjGetSweeps,
    cusolverDnHandle_t, handle,
    syevjInfo_t, info,
    int*, executed_sweeps)


DEF_FN(cusolverStatus_t, cusolverDnSsyevjBatched_bufferSize,
    cusolverDnHandle_t, handle,
    cusolverEigMode_t, jobz, 
    cublasFillMode_t, uplo, 
    int, n,
    const float*, A,
    int, lda,
    const float*, W,
    int*, lwork,
    syevjInfo_t, params,
    int batchSize
    )

DEF_FN(cusolverStatus_t, cusolverDnDsyevjBatched_bufferSize,
    cusolverDnHandle_t, handle,
    cusolverEigMode_t, jobz, 
    cublasFillMode_t, uplo, 
    int, n,
    const double*, A, 
    int, lda,
    const double*, W,
    int*, lwork,
    syevjInfo_t, params,
    int batchSize
    )

DEF_FN(cusolverStatus_t, cusolverDnCheevjBatched_bufferSize,
    cusolverDnHandle_t, handle,
    cusolverEigMode_t, jobz, 
    cublasFillMode_t, uplo, 
    int, n,
    const cuComplex*, A, 
    int, lda,
    const float*, W,
    int*, lwork,
    syevjInfo_t, params,
    int batchSize
    )

DEF_FN(cusolverStatus_t, cusolverDnZheevjBatched_bufferSize,
    cusolverDnHandle_t, handle,
    cusolverEigMode_t, jobz, 
    cublasFillMode_t, uplo, 
    int, n,
    const cuDoubleComplex*, A, 
    int, lda,
    const double*, W,
    int*, lwork,
    syevjInfo_t, params,
    int batchSize
    )


DEF_FN(cusolverStatus_t, cusolverDnSsyevjBatched,
    cusolverDnHandle_t, handle,
    cusolverEigMode_t, jobz,
    cublasFillMode_t, uplo,
    int, n,   
    float*, A,
    int, lda,
    float*, W, 
    float*, work,
    int, lwork,
    int*, info, 
    syevjInfo_t, params,
    int batchSize
    )

DEF_FN(cusolverStatus_t, cusolverDnDsyevjBatched,
    cusolverDnHandle_t, handle,
    cusolverEigMode_t, jobz,
    cublasFillMode_t, uplo, 
    int, n,
    double*, A,
    int, lda,
    double*, W,
    double*, work,
    int, lwork,
    int*, info,
    syevjInfo_t, params,
    int batchSize
    )

DEF_FN(cusolverStatus_t, cusolverDnCheevjBatched,
    cusolverDnHandle_t, handle,
    cusolverEigMode_t, jobz,
    cublasFillMode_t, uplo, 
    int, n,
    cuComplex*, A,
    int, lda,
    float*, W,
    cuComplex*, work,
    int, lwork,
    int*, info,
    syevjInfo_t, params,
    int batchSize
    )

DEF_FN(cusolverStatus_t, cusolverDnZheevjBatched,
    cusolverDnHandle_t, handle,
    cusolverEigMode_t, jobz, 
    cublasFillMode_t, uplo, 
    int, n,
    cuDoubleComplex*, A,
    int, lda,
    double*, W,
    cuDoubleComplex*, work,
    int, lwork,
    int*, info,
    syevjInfo_t, params,
    int batchSize
    )


DEF_FN(cusolverStatus_t, cusolverDnSsyevj_bufferSize,
    cusolverDnHandle_t, handle,
    cusolverEigMode_t, jobz,
    cublasFillMode_t, uplo,
    int, n,
    const float*, A,
    int, lda,
    const float*, W,
    int*, lwork,
    syevjInfo_t, params)


DEF_FN(cusolverStatus_t, cusolverDnDsyevj_bufferSize,
    cusolverDnHandle_t, handle,
    cusolverEigMode_t, jobz,
    cublasFillMode_t, uplo,
    int, n,
    const double*, A,
    int, lda,
    const double*, W,
    int*, lwork,
    syevjInfo_t, params)


DEF_FN(cusolverStatus_t, cusolverDnCheevj_bufferSize,
    cusolverDnHandle_t, handle,
    cusolverEigMode_t, jobz,
    cublasFillMode_t, uplo,
    int, n,
    const cuComplex*, A,
    int, lda,
    const float*, W,
    int*, lwork,
    syevjInfo_t, params)


DEF_FN(cusolverStatus_t, cusolverDnZheevj_bufferSize,
    cusolverDnHandle_t, handle,
    cusolverEigMode_t, jobz, 
    cublasFillMode_t, uplo,
    int, n,
    const cuDoubleComplex*, A,
    int, lda,
    const double*, W,
    int*, lwork,
    syevjInfo_t, params)


DEF_FN(cusolverStatus_t, cusolverDnSsyevj,
    cusolverDnHandle_t, handle,
    cusolverEigMode_t, jobz, 
    cublasFillMode_t, uplo, 
    int, n,
    float*, A,
    int, lda,
    float*, W,
    float*, work,
    int, lwork,
    int*, info,
    syevjInfo_t, params)


DEF_FN(cusolverStatus_t, cusolverDnDsyevj,
    cusolverDnHandle_t, handle,
    cusolverEigMode_t, jobz, 
    cublasFillMode_t, uplo,
    int, n,
    double*, A,
    int, lda,
    double*, W, 
    double*, work,
    int, lwork,
    int*, info,
    syevjInfo_t, params)


DEF_FN(cusolverStatus_t, cusolverDnCheevj,
    cusolverDnHandle_t, handle,
    cusolverEigMode_t, jobz, 
    cublasFillMode_t, uplo,
    int, n,
    cuComplex*, A,
    int, lda,
    float*, W, 
    cuComplex*, work,
    int, lwork,
    int*, info,
    syevjInfo_t, params)


DEF_FN(cusolverStatus_t, cusolverDnZheevj,
    cusolverDnHandle_t, handle,
    cusolverEigMode_t, jobz,
    cublasFillMode_t, uplo,
    int, n,
    cuDoubleComplex*, A,
    int, lda,
    double*, W, 
    cuDoubleComplex*, work,
    int, lwork,
    int*, info,
    syevjInfo_t, params)


DEF_FN(cusolverStatus_t, cusolverDnSsygvj_bufferSize,
    cusolverDnHandle_t, handle,
    cusolverEigType_t, itype, 
    cusolverEigMode_t, jobz, 
    cublasFillMode_t, uplo,
    int, n,
    const float*, A, 
    int, lda,
    const float*, B, 
    int, ldb,
    const float*, W,
    int*, lwork,
    syevjInfo_t, params)

DEF_FN(cusolverStatus_t, cusolverDnDsygvj_bufferSize,
    cusolverDnHandle_t, handle,
    cusolverEigType_t, itype, 
    cusolverEigMode_t, jobz, 
    cublasFillMode_t, uplo,
    int, n,
    const double*, A, 
    int, lda,
    const double*, B,
    int, ldb,
    const double*, W,
    int*, lwork,
    syevjInfo_t, params)

DEF_FN(cusolverStatus_t, cusolverDnChegvj_bufferSize,
    cusolverDnHandle_t, handle,
    cusolverEigType_t, itype,
    cusolverEigMode_t, jobz,
    cublasFillMode_t, uplo,
    int, n,
    const cuComplex*, A, 
    int, lda,
    const cuComplex*, B, 
    int, ldb,
    const float*, W,
    int*, lwork,
    syevjInfo_t, params)

DEF_FN(cusolverStatus_t, cusolverDnZhegvj_bufferSize,
    cusolverDnHandle_t, handle,
    cusolverEigType_t, itype,
    cusolverEigMode_t, jobz,
    cublasFillMode_t, uplo,
    int, n,
    const cuDoubleComplex*, A, 
    int, lda,
    const cuDoubleComplex*, B, 
    int, ldb,
    const double*, W,
    int*, lwork,
    syevjInfo_t, params)

DEF_FN(cusolverStatus_t, cusolverDnSsygvj,
    cusolverDnHandle_t, handle,
    cusolverEigType_t, itype, 
    cusolverEigMode_t, jobz,
    cublasFillMode_t, uplo, 
    int, n,
    float*, A, 
    int, lda,
    float*, B, 
    int, ldb,
    float*, W,
    float*, work,
    int, lwork,
    int*, info,
    syevjInfo_t, params)

DEF_FN(cusolverStatus_t, cusolverDnDsygvj,
    cusolverDnHandle_t, handle,
    cusolverEigType_t, itype, 
    cusolverEigMode_t, jobz,
    cublasFillMode_t, uplo, 
    int, n,
    double*, A, 
    int, lda,
    double*, B,
    int, ldb,
    double*, W, 
    double*, work,
    int, lwork,
    int*, info,
    syevjInfo_t, params)

DEF_FN(cusolverStatus_t, cusolverDnChegvj,
    cusolverDnHandle_t, handle,
    cusolverEigType_t, itype, 
    cusolverEigMode_t, jobz, 
    cublasFillMode_t, uplo,
    int, n,
    cuComplex*, A, 
    int, lda,
    cuComplex*, B, 
    int, ldb,
    float*, W,
    cuComplex*, work,
    int, lwork,
    int*, info,
    syevjInfo_t, params)

DEF_FN(cusolverStatus_t, cusolverDnZhegvj,
    cusolverDnHandle_t, handle,
    cusolverEigType_t, itype, 
    cusolverEigMode_t, jobz,
    cublasFillMode_t, uplo,  
    int, n,
    cuDoubleComplex*, A, 
    int, lda,
    cuDoubleComplex*, B, 
    int, ldb,
    double*, W, 
    cuDoubleComplex*, work,
    int, lwork,
    int*, info,
    syevjInfo_t, params)


DEF_FN(cusolverStatus_t, cusolverDnCreateGesvdjInfo,
    gesvdjInfo_t*, info)

DEF_FN(cusolverStatus_t, cusolverDnDestroyGesvdjInfo,
    gesvdjInfo_t, info)

DEF_FN(cusolverStatus_t, cusolverDnXgesvdjSetTolerance,
    gesvdjInfo_t, info,
    double, tolerance)

DEF_FN(cusolverStatus_t, cusolverDnXgesvdjSetMaxSweeps,
    gesvdjInfo_t, info,
    int, max_sweeps)

DEF_FN(cusolverStatus_t, cusolverDnXgesvdjSetSortEig,
    gesvdjInfo_t, info,
    int, sort_svd)

DEF_FN(cusolverStatus_t, cusolverDnXgesvdjGetResidual,
    cusolverDnHandle_t, handle,
    gesvdjInfo_t, info,
    double*, residual)

DEF_FN(cusolverStatus_t, cusolverDnXgesvdjGetSweeps,
    cusolverDnHandle_t, handle,
    gesvdjInfo_t, info,
    int*, executed_sweeps)

DEF_FN(cusolverStatus_t, cusolverDnSgesvdjBatched_bufferSize,
    cusolverDnHandle_t, handle,
    cusolverEigMode_t, jobz,
    int, m,                
    int, n,                
    const float*, A,    
    int, lda,           
    const float*, S, 
    const float*, U,   
    int, ldu, 
    const float*, V,
    int, ldv,  
    int*, lwork,
    gesvdjInfo_t, params,
    int, batchSize)

DEF_FN(cusolverStatus_t, cusolverDnDgesvdjBatched_bufferSize,
    cusolverDnHandle_t, handle,
    cusolverEigMode_t, jobz,
    int, m,
    int, n,
    const double*, A, 
    int, lda,
    const double*, S,
    const double*, U,
    int, ldu,
    const double*, V,
    int, ldv,
    int*, lwork,
    gesvdjInfo_t, params,
    int, batchSize)

DEF_FN(cusolverStatus_t, cusolverDnCgesvdjBatched_bufferSize,
    cusolverDnHandle_t, handle,
    cusolverEigMode_t, jobz,
    int, m,
    int, n,
    const cuComplex*, A,
    int, lda,
    const float*, S,
    const cuComplex*, U,
    int, ldu,
    const cuComplex*, V,
    int, ldv,
    int*, lwork,
    gesvdjInfo_t, params,
    int, batchSize)

DEF_FN(cusolverStatus_t, cusolverDnZgesvdjBatched_bufferSize,
    cusolverDnHandle_t, handle,
    cusolverEigMode_t, jobz, 
    int, m, 
    int, n, 
    const cuDoubleComplex*, A,
    int, lda,
    const double*, S,
    const cuDoubleComplex*, U,
    int, ldu, 
    const cuDoubleComplex*, V,
    int, ldv,
    int*, lwork,
    gesvdjInfo_t, params,
    int, batchSize)

DEF_FN(cusolverStatus_t, cusolverDnSgesvdjBatched,
    cusolverDnHandle_t, handle,
    cusolverEigMode_t, jobz, 
    int, m, 
    int, n, 
    float*, A, 
    int, lda, 
    float*, S, 
    float*, U,
    int, ldu,
    float*, V,
    int, ldv, 
    float*, work,
    int, lwork,
    int*, info,
    gesvdjInfo_t, params,
    int, batchSize)

DEF_FN(cusolverStatus_t, cusolverDnDgesvdjBatched,
    cusolverDnHandle_t, handle,
    cusolverEigMode_t, jobz,
    int, m,
    int, n,
    double*, A,
    int, lda,
    double*, S,
    double*, U,
    int, ldu,
    double*, V,
    int, ldv, 
    double*, work,
    int, lwork,
    int*, info, 
    gesvdjInfo_t, params,
    int, batchSize)

DEF_FN(cusolverStatus_t, cusolverDnCgesvdjBatched,
    cusolverDnHandle_t, handle,
    cusolverEigMode_t, jobz,
    int, m, 
    int, n,
    cuComplex*, A,
    int, lda,
    float*, S,
    cuComplex*, U,
    int, ldu,
    cuComplex*, V,
    int, ldv,
    cuComplex*, work,
    int, lwork,
    int*, info,
    gesvdjInfo_t, params,
    int, batchSize)

DEF_FN(cusolverStatus_t, cusolverDnZgesvdjBatched,
    cusolverDnHandle_t, handle,
    cusolverEigMode_t, jobz,
    int, m,
    int, n,
    cuDoubleComplex*, A,
    int, lda, 
    double*, S, 
    cuDoubleComplex*, U,
    int, ldu,
    cuDoubleComplex*, V,
    int, ldv,
    cuDoubleComplex*, work,
    int, lwork,
    int*, info,
    gesvdjInfo_t, params,
    int, batchSize)

DEF_FN(cusolverStatus_t, cusolverDnSgesvdj_bufferSize,
    cusolverDnHandle_t, handle,
    cusolverEigMode_t, jobz, 
    int, econ,
    int, m,
    int, n, 
    const float*, A,
    int, lda,
    const float*, S,
    const float*, U,
    int, ldu, 
    const float*, V,
    int, ldv,
    int*, lwork,
    gesvdjInfo_t, params)

DEF_FN(cusolverStatus_t, cusolverDnDgesvdj_bufferSize,
    cusolverDnHandle_t, handle,
    cusolverEigMode_t, jobz, 
    int, econ,
    int, m,
    int, n,
    const double*, A, 
    int, lda,
    const double*, S,
    const double*, U,
    int, ldu,
    const double*, V,
    int, ldv,
    int*, lwork,
    gesvdjInfo_t, params)

DEF_FN(cusolverStatus_t, cusolverDnCgesvdj_bufferSize,
    cusolverDnHandle_t, handle,
    cusolverEigMode_t, jobz, 
    int, econ,
    int, m,
    int, n,
    const cuComplex*, A,
    int, lda,
    const float*, S,
    const cuComplex*, U,
    int, ldu,
    const cuComplex*, V,
    int, ldv,
    int*, lwork,
    gesvdjInfo_t, params)

DEF_FN(cusolverStatus_t, cusolverDnZgesvdj_bufferSize,
    cusolverDnHandle_t, handle,
    cusolverEigMode_t, jobz,
    int, econ,
    int, m,
    int, n,
    const cuDoubleComplex*, A,
    int, lda,
    const double*, S,
    const cuDoubleComplex*, U,
    int, ldu,
    const cuDoubleComplex*, V,
    int, ldv, 
    int*, lwork,
    gesvdjInfo_t, params)

DEF_FN(cusolverStatus_t, cusolverDnSgesvdj,
    cusolverDnHandle_t, handle,
    cusolverEigMode_t, jobz,
    int, econ,
    int, m,
    int, n,
    float*, A, 
    int, lda,
    float*, S,
    float*, U,
    int, ldu,
    float*, V,
    int, ldv,
    float*, work,
    int, lwork,
    int*, info,
    gesvdjInfo_t, params)

DEF_FN(cusolverStatus_t, cusolverDnDgesvdj,
    cusolverDnHandle_t, handle,
    cusolverEigMode_t, jobz,
    int, econ, 
    int, m, 
    int, n, 
    double*, A, 
    int, lda,
    double*, S,
    double*, U,
    int, ldu,
    double*, V,
    int, ldv,
    double*, work,
    int, lwork,
    int*, info,
    gesvdjInfo_t, params)

DEF_FN(cusolverStatus_t, cusolverDnCgesvdj,
    cusolverDnHandle_t, handle,
    cusolverEigMode_t, jobz,
    int, econ,
    int, m,
    int, n,
    cuComplex*, A,
    int, lda,
    float*, S,
    cuComplex*, U,
    int, ldu,
    cuComplex*, V,
    int, ldv,
    cuComplex*, work,
    int, lwork,
    int*, info,
    gesvdjInfo_t, params)

DEF_FN(cusolverStatus_t, cusolverDnZgesvdj,
    cusolverDnHandle_t, handle,
    cusolverEigMode_t, jobz,
    int, econ,
    int, m,
    int, n,
    cuDoubleComplex*, A,
    int, lda,
    double*, S,
    cuDoubleComplex*, U, 
    int, ldu, 
    cuDoubleComplex*, V,
    int, ldv,
    cuDoubleComplex*, work,
    int, lwork,
    int*, info,
    gesvdjInfo_t, params)



DEF_FN(cusolverStatus_t, cusolverDnSgesvdaStridedBatched_bufferSize,
    cusolverDnHandle_t, handle,
    cusolverEigMode_t, jobz,
    int, rank,
    int, m,
    int, n,
    const float*, d_A, 
    int, lda,
    long long int, strideA, 
    const float*, d_S, 
    long long int, strideS, 
    const float*, d_U, 
    int, ldu,
    long long int, strideU, 
    const float*, d_V, 
    int, ldv,
    long long int, strideV,
    int*, lwork,
    int batchSize
    )


DEF_FN(cusolverStatus_t, cusolverDnDgesvdaStridedBatched_bufferSize,
    cusolverDnHandle_t, handle,
    cusolverEigMode_t, jobz,
    int, rank,
    int, m,
    int, n,
    const double*, d_A, 
    int, lda,
    long long int, strideA, 
    const double*, d_S,   
    long long int, strideS, 
    const double*, d_U,  
    int, ldu,
    long long int, strideU, 
    const double*, d_V,
    int, ldv,
    long long int, strideV, 
    int*, lwork,
    int batchSize
    )


DEF_FN(cusolverStatus_t, cusolverDnCgesvdaStridedBatched_bufferSize,
    cusolverDnHandle_t, handle,
    cusolverEigMode_t, jobz,
    int, rank,
    int, m,
    int, n,
    const cuComplex*, d_A, 
    int, lda,
    long long int, strideA, 
    const float*, d_S, 
    long long int, strideS, 
    const cuComplex*, d_U,
    int, ldu,
    long long int, strideU, 
    const cuComplex*, d_V, 
    int, ldv,
    long long int, strideV, 
    int*, lwork,
    int batchSize
    )

DEF_FN(cusolverStatus_t, cusolverDnZgesvdaStridedBatched_bufferSize,
    cusolverDnHandle_t, handle,
    cusolverEigMode_t, jobz,
    int, rank,
    int, m,
    int, n,
    const cuDoubleComplex*, d_A,
    int, lda,
    long long int, strideA,
    const double*, d_S, 
    long long int, strideS, 
    const cuDoubleComplex*, d_U, 
    int, ldu,
    long long int, strideU,
    const cuDoubleComplex*, d_V,
    int, ldv,
    long long int, strideV, 
    int*, lwork,
    int batchSize
    )


DEF_FN(cusolverStatus_t, cusolverDnSgesvdaStridedBatched,
    cusolverDnHandle_t, handle,
    cusolverEigMode_t, jobz, 
    int, rank, 
    int, m,   
    int, n,  
    const float*, d_A, 
    int, lda, 
    long long int, strideA,
    float*, d_S, 
    long long int, strideS, 
    float*, d_U, 
    int, ldu, 
    long long int, strideU,
    float*, d_V, 
    int, ldv,    
    long long int, strideV, 
    float*, d_work,
    int, lwork,
    int*, d_info,
    double*, h_R_nrmF,
    int, batchSize)


DEF_FN(cusolverStatus_t, cusolverDnDgesvdaStridedBatched,
    cusolverDnHandle_t, handle,
    cusolverEigMode_t, jobz, 
    int, rank,
    int, m, 
    int, n, 
    const double*, d_A,
    int, lda,  
    long long int, strideA, 
    double*, d_S, 
    long long int, strideS,
    double*, d_U, 
    int, ldu, 
    long long int, strideU, 
    double*, d_V, 
    int, ldv, 
    long long int, strideV,
    double*, d_work,
    int, lwork,
    int*, d_info,
    double*, h_R_nrmF, 
    int, batchSize)


DEF_FN(cusolverStatus_t, cusolverDnCgesvdaStridedBatched,
    cusolverDnHandle_t, handle,
    cusolverEigMode_t, jobz, 
    int, rank,  
    int, m, 
    int, n, 
    const cuComplex*, d_A, 
    int, lda,
    long long int, strideA,
    float*, d_S,
    long long int, strideS,
    cuComplex*, d_U, 
    int, ldu,   
    long long int, strideU,  
    cuComplex*, d_V, 
    int, ldv, 
    long long int, strideV,
    cuComplex*, d_work,
    int, lwork,
    int*, d_info,
    double*, h_R_nrmF, 
    int, batchSize)


DEF_FN(cusolverStatus_t, cusolverDnZgesvdaStridedBatched,
    cusolverDnHandle_t, handle,
    cusolverEigMode_t, jobz, 
    int, rank, 
    int, m,   
    int, n,  
    const cuDoubleComplex*, d_A, 
    int, lda,    
    long long int, strideA,
    double*, d_S,
    long long int, strideS,
    cuDoubleComplex*, d_U, 
    int, ldu,   
    long long int, strideU, 
    cuDoubleComplex*, d_V,
    int, ldv, 
    long long int, strideV, 
    cuDoubleComplex*, d_work,
    int, lwork,
    int*, d_info,
    double*, h_R_nrmF,
    int, batchSize)


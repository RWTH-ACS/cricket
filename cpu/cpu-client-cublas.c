
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

cublasStatus_t cublasCreate_v2(cublasHandle_t* handle)
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
        *handle = (cublasHandle_t)result.ptr_result_u.ptr;
    }
    return result.err;
}

cublasStatus_t cublasDestroy_v2(cublasHandle_t handle)
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

DEF_FN(cublasStatus_t, cublasGetVersion_v2, cublasHandle_t, handle, int*, version);
DEF_FN(cublasStatus_t, cublasGetProperty, libraryPropertyType, type, int*, value);
DEF_FN(size_t, cublasGetCudartVersion);
cublasStatus_t cublasSetWorkspace_v2(cublasHandle_t handle, void* workspace, size_t workspaceSizeInBytes)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval_1;
    retval_1 = rpc_cublassetworkspace_1(
        (ptr)handle,
        (ptr)workspace,
        workspaceSizeInBytes,
        &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    return result;
}

cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, cudaStream_t streamId)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval_1;
    retval_1 = rpc_cublassetstream_1(
        (ptr)handle,
        (ptr)streamId,
        &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    return result;
}

DEF_FN(cublasStatus_t, cublasGetStream_v2, cublasHandle_t, handle, cudaStream_t*, streamId);
DEF_FN(cublasStatus_t, cublasGetPointerMode_v2, cublasHandle_t, handle, cublasPointerMode_t*, mode);
DEF_FN(cublasStatus_t, cublasSetPointerMode_v2, cublasHandle_t, handle, cublasPointerMode_t, mode);
DEF_FN(cublasStatus_t, cublasGetAtomicsMode, cublasHandle_t, handle, cublasAtomicsMode_t*, mode);
DEF_FN(cublasStatus_t, cublasSetAtomicsMode, cublasHandle_t, handle, cublasAtomicsMode_t, mode);

cublasStatus_t cublasGetMathMode(cublasHandle_t handle, cublasMath_t *mode)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT

    int_result result;
    enum clnt_stat retval_1;
    retval_1 = rpc_cublasgetmathmode_1(
        (ptr)handle,
        &result, clnt
    );
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    if (result.err == 0) {
        *mode = result.int_result_u.data;
    }
    return result.err;
}

cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval_1;
    retval_1 = rpc_cublassetmathmode_1(
        (ptr)handle,
        (int)mode,
        &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    return result;
}

DEF_FN(cublasStatus_t, cublasGetSmCountTarget, cublasHandle_t, handle, int*, smCountTarget);
DEF_FN(cublasStatus_t, cublasSetSmCountTarget, cublasHandle_t, handle, int, smCountTarget);
DEF_FN(const char*, cublasGetStatusName, cublasStatus_t, status);
DEF_FN(const char*, cublasGetStatusString, cublasStatus_t, status);
DEF_FN(cublasStatus_t, cublasLoggerConfigure, int, logIsOn, int, logToStdOut, int, logToStdErr, const char*, logFileName);
DEF_FN(cublasStatus_t, cublasSetLoggerCallback, cublasLogCallback, userCallback);
DEF_FN(cublasStatus_t, cublasGetLoggerCallback, cublasLogCallback*, userCallback);
DEF_FN(cublasStatus_t, cublasSetVector, int, n, int, elemSize, const void*, x, int, incx, void*, devicePtr, int, incy);
DEF_FN(cublasStatus_t, cublasSetVector_64, int64_t, n, int64_t, elemSize, const void*, x, int64_t, incx, void*, devicePtr, int64_t, incy);
DEF_FN(cublasStatus_t, cublasGetVector, int, n, int, elemSize, const void*, x, int, incx, void*, y, int, incy);
DEF_FN(cublasStatus_t, cublasGetVector_64, int64_t, n, int64_t, elemSize, const void*, x, int64_t, incx, void*, y, int64_t, incy);
DEF_FN(cublasStatus_t, cublasSetMatrix, int, rows, int, cols, int, elemSize, const void*, A, int, lda, void*, B, int, ldb);
DEF_FN(cublasStatus_t, cublasSetMatrix_64, int64_t, rows, int64_t, cols, int64_t, elemSize, const void*, A, int64_t, lda, void*, B, int64_t, ldb);
DEF_FN(cublasStatus_t, cublasGetMatrix, int, rows, int, cols, int, elemSize, const void*, A, int, lda, void*, B, int, ldb);
DEF_FN(cublasStatus_t, cublasGetMatrix_64, int64_t, rows, int64_t, cols, int64_t, elemSize, const void*, A, int64_t, lda, void*, B, int64_t, ldb);
DEF_FN(cublasStatus_t, cublasSetVectorAsync, , int, n, int, elemSize, const void*, hostPtr, int, incx, void*, devicePtr, int, incy, cudaStream_t, stream);
DEF_FN(cublasStatus_t, cublasSetVectorAsync_64, , int64_t, n, int64_t, elemSize, const void*, hostPtr, int64_t, incx, void*, devicePtr, int64_t, incy, cudaStream_t, stream);
DEF_FN(cublasStatus_t, cublasGetVectorAsync, , int, n, int, elemSize, const void*, devicePtr, int, incx, void*, hostPtr, int, incy, cudaStream_t, stream);
DEF_FN(cublasStatus_t, cublasGetVectorAsync_64, , int64_t, n, int64_t, elemSize, const void*, devicePtr, int64_t, incx, void*, hostPtr, int64_t, incy, cudaStream_t, stream);
DEF_FN(cublasStatus_t, cublasSetMatrixAsync, int, rows, int, cols, int, elemSize, const void*, A, int, lda, void*, B, int, ldb, cudaStream_t, stream);
DEF_FN(cublasStatus_t, cublasSetMatrixAsync_64, int64_t, rows, int64_t, cols, int64_t, elemSize, const void*, A, int64_t, lda, void*, B, int64_t, ldb, cudaStream_t, stream);
DEF_FN(cublasStatus_t, cublasGetMatrixAsync, int, rows, int, cols, int, elemSize, const void*, A, int, lda, void*, B, int, ldb, cudaStream_t, stream);
DEF_FN(cublasStatus_t, cublasGetMatrixAsync_64, int64_t, rows, int64_t, cols, int64_t, elemSize, const void*, A, int64_t, lda, void*, B, int64_t, ldb, cudaStream_t, stream);
void cublasXerbla(const char* srName, int info) {
    void (*fun)(const char*, int);
    char* error_str; *(void **)(&fun) = dlsym(libwrap_get_sohandle(), "cublasXerbla");
    if ((error_str = dlerror()) != ((void *)0)) {
        if (0 > get_log_data()->curr_level) ;
        else 
            loggfe(0, 88, "/home/eiling/projects/cricket/cpu/cpu-client-cublas.c", "[libwrap] %s", error_str); 
    }
    if (3 > get_log_data()->curr_level) ;
    else 
        loggf(3, "%s called", "cublasXerbla");
    (*fun)(srName, info); 
    if (3 > get_log_data()->curr_level) ;
    else loggf(3, "%s finished", "cublasXerbla");
}
DEF_FN(cublasStatus_t, cublasNrm2Ex, cublasHandle_t, handle, int, n, const void*, x, cudaDataType, xType, int, incx, void*, result, cudaDataType, resultType, cudaDataType, executionType);
DEF_FN(cublasStatus_t, cublasNrm2Ex_64, cublasHandle_t, handle, int64_t, n, const void*, x, cudaDataType, xType, int64_t, incx, void*, result, cudaDataType, resultType, cudaDataType, executionType);
DEF_FN(cublasStatus_t, cublasSnrm2_v2, cublasHandle_t, handle, int, n, const float*, x, int, incx, float*, result);
DEF_FN(cublasStatus_t, cublasSnrm2_v2_64, cublasHandle_t, handle, int64_t, n, const float*, x, int64_t, incx, float*, result);
DEF_FN(cublasStatus_t, cublasDnrm2_v2, cublasHandle_t, handle, int, n, const double*, x, int, incx, double*, result);
DEF_FN(cublasStatus_t, cublasDnrm2_v2_64, cublasHandle_t, handle, int64_t, n, const double*, x, int64_t, incx, double*, result);
DEF_FN(cublasStatus_t, cublasScnrm2_v2, cublasHandle_t, handle, int, n, const cuComplex*, x, int, incx, float*, result);
DEF_FN(cublasStatus_t, cublasScnrm2_v2_64, cublasHandle_t, handle, int64_t, n, const cuComplex*, x, int64_t, incx, float*, result);
DEF_FN(cublasStatus_t, cublasDznrm2_v2, cublasHandle_t, handle, int, n, const cuDoubleComplex*, x, int, incx, double*, result);
DEF_FN(cublasStatus_t, cublasDznrm2_v2_64, cublasHandle_t, handle, int64_t, n, const cuDoubleComplex*, x, int64_t, incx, double*, result);
DEF_FN(cublasStatus_t, cublasDotEx, cublasHandle_t, handle, int, n, const void*, x, cudaDataType, xType, int, incx, const void*, y, cudaDataType, yType, int, incy, void*, result, cudaDataType, resultType, cudaDataType, executionType);
DEF_FN(cublasStatus_t, cublasDotEx_64, cublasHandle_t, handle, int64_t, n, const void*, x, cudaDataType, xType, int64_t, incx, const void*, y, cudaDataType, yType, int64_t, incy, void*, result, cudaDataType, resultType, cudaDataType, executionType);
DEF_FN(cublasStatus_t, cublasDotcEx, cublasHandle_t, handle, int, n, const void*, x, cudaDataType, xType, int, incx, const void*, y, cudaDataType, yType, int, incy, void*, result, cudaDataType, resultType, cudaDataType, executionType);
DEF_FN(cublasStatus_t, cublasDotcEx_64, cublasHandle_t, handle, int64_t, n, const void*, x, cudaDataType, xType, int64_t, incx, const void*, y, cudaDataType, yType, int64_t, incy, void*, result, cudaDataType, resultType, cudaDataType, executionType);
DEF_FN(cublasStatus_t, cublasSdot_v2, cublasHandle_t, handle, int, n, const float*, x, int, incx, const float*, y, int, incy, float*, result);
DEF_FN(cublasStatus_t, cublasSdot_v2_64, , cublasHandle_t, handle, int64_t, n, const float*, x, int64_t, incx, const float*, y, int64_t, incy, float*, result);
DEF_FN(cublasStatus_t, cublasDdot_v2, cublasHandle_t, handle, int, n, const double*, x, int, incx, const double*, y, int, incy, double*, result);
DEF_FN(cublasStatus_t, cublasDdot_v2_64, , cublasHandle_t, handle, int64_t, n, const double*, x, int64_t, incx, const double*, y, int64_t, incy, double*, result);
DEF_FN(cublasStatus_t, cublasCdotu_v2, , cublasHandle_t, handle, int, n, const cuComplex*, x, int, incx, const cuComplex*, y, int, incy, cuComplex*, result);
DEF_FN(cublasStatus_t, cublasCdotu_v2_64, cublasHandle_t, handle, int64_t, n, const cuComplex*, x, int64_t, incx, const cuComplex*, y, int64_t, incy, cuComplex*, result);
DEF_FN(cublasStatus_t, cublasCdotc_v2, , cublasHandle_t, handle, int, n, const cuComplex*, x, int, incx, const cuComplex*, y, int, incy, cuComplex*, result);
DEF_FN(cublasStatus_t, cublasCdotc_v2_64, cublasHandle_t, handle, int64_t, n, const cuComplex*, x, int64_t, incx, const cuComplex*, y, int64_t, incy, cuComplex*, result);
DEF_FN(cublasStatus_t, cublasZdotu_v2, cublasHandle_t, handle, int, n, const cuDoubleComplex*, x, int, incx, const cuDoubleComplex*, y, int, incy, cuDoubleComplex*, result);
DEF_FN(cublasStatus_t, cublasZdotu_v2_64, cublasHandle_t, handle, int64_t, n, const cuDoubleComplex*, x, int64_t, incx, const cuDoubleComplex*, y, int64_t, incy, cuDoubleComplex*, result);
DEF_FN(cublasStatus_t, cublasZdotc_v2, cublasHandle_t, handle, int, n, const cuDoubleComplex*, x, int, incx, const cuDoubleComplex*, y, int, incy, cuDoubleComplex*, result);
DEF_FN(cublasStatus_t, cublasZdotc_v2_64, cublasHandle_t, handle, int64_t, n, const cuDoubleComplex*, x, int64_t, incx, const cuDoubleComplex*, y, int64_t, incy, cuDoubleComplex*, result);
DEF_FN(cublasStatus_t, cublasScalEx, cublasHandle_t, handle, int, n, const void*, alpha, cudaDataType, alphaType, void*, x, cudaDataType, xType, int, incx, cudaDataType, executionType);
DEF_FN(cublasStatus_t, cublasScalEx_64, cublasHandle_t, handle, int64_t, n, const void*, alpha, cudaDataType, alphaType, void*, x, cudaDataType, xType, int64_t, incx, cudaDataType, executionType);
DEF_FN(cublasStatus_t, cublasSscal_v2, cublasHandle_t, handle, int, n, const float*, alpha, float*, x, int, incx);
DEF_FN(cublasStatus_t, cublasSscal_v2_64, cublasHandle_t, handle, int64_t, n, const float*, alpha, float*, x, int64_t, incx);
DEF_FN(cublasStatus_t, cublasDscal_v2, cublasHandle_t, handle, int, n, const double*, alpha, double*, x, int, incx);
DEF_FN(cublasStatus_t, cublasDscal_v2_64, cublasHandle_t, handle, int64_t, n, const double*, alpha, double*, x, int64_t, incx);
DEF_FN(cublasStatus_t, cublasCscal_v2, cublasHandle_t, handle, int, n, const cuComplex*, alpha, cuComplex*, x, int, incx);
DEF_FN(cublasStatus_t, cublasCscal_v2_64, cublasHandle_t, handle, int64_t, n, const cuComplex*, alpha, cuComplex*, x, int64_t, incx);
DEF_FN(cublasStatus_t, cublasCsscal_v2, cublasHandle_t, handle, int, n, const float*, alpha, cuComplex*, x, int, incx);
DEF_FN(cublasStatus_t, cublasCsscal_v2_64, cublasHandle_t, handle, int64_t, n, const float*, alpha, cuComplex*, x, int64_t, incx);
DEF_FN(cublasStatus_t, cublasZscal_v2, cublasHandle_t, handle, int, n, const cuDoubleComplex*, alpha, cuDoubleComplex*, x, int, incx);
DEF_FN(cublasStatus_t, cublasZscal_v2_64, cublasHandle_t, handle, int64_t, n, const cuDoubleComplex*, alpha, cuDoubleComplex*, x, int64_t, incx);
DEF_FN(cublasStatus_t, cublasZdscal_v2, cublasHandle_t, handle, int, n, const double*, alpha, cuDoubleComplex*, x, int, incx);
DEF_FN(cublasStatus_t, cublasZdscal_v2_64, cublasHandle_t, handle, int64_t, n, const double*, alpha, cuDoubleComplex*, x, int64_t, incx);
DEF_FN(cublasStatus_t, cublasAxpyEx, cublasHandle_t, handle, int, n, const void*, alpha, cudaDataType, alphaType, const void*, x, cudaDataType, xType, int, incx, void*, y, cudaDataType, yType, int, incy, cudaDataType, executiontype);
DEF_FN(cublasStatus_t, cublasAxpyEx_64, cublasHandle_t, handle, int64_t, n, const void*, alpha, cudaDataType, alphaType, const void*, x, cudaDataType, xType, int64_t, incx, void*, y, cudaDataType, yType, int64_t, incy, cudaDataType, executiontype);
DEF_FN(cublasStatus_t, cublasSaxpy_v2, cublasHandle_t, handle, int, n, const float*, alpha, const float*, x, int, incx, float*, y, int, incy);
DEF_FN(cublasStatus_t, cublasSaxpy_v2_64, , cublasHandle_t, handle, int64_t, n, const float*, alpha, const float*, x, int64_t, incx, float*, y, int64_t, incy);
DEF_FN(cublasStatus_t, cublasDaxpy_v2, cublasHandle_t, handle, int, n, const double*, alpha, const double*, x, int, incx, double*, y, int, incy);
DEF_FN(cublasStatus_t, cublasDaxpy_v2_64, , cublasHandle_t, handle, int64_t, n, const double*, alpha, const double*, x, int64_t, incx, double*, y, int64_t, incy);
DEF_FN(cublasStatus_t, cublasCaxpy_v2, , cublasHandle_t, handle, int, n, const cuComplex*, alpha, const cuComplex*, x, int, incx, cuComplex*, y, int, incy);
DEF_FN(cublasStatus_t, cublasCaxpy_v2_64, cublasHandle_t, handle, int64_t, n, const cuComplex*, alpha, const cuComplex*, x, int64_t, incx, cuComplex*, y, int64_t, incy);
DEF_FN(cublasStatus_t, cublasZaxpy_v2, cublasHandle_t, handle, int, n, const cuDoubleComplex*, alpha, const cuDoubleComplex*, x, int, incx, cuDoubleComplex*, y, int, incy);
DEF_FN(cublasStatus_t, cublasZaxpy_v2_64, cublasHandle_t, handle, int64_t, n, const cuDoubleComplex*, alpha, const cuDoubleComplex*, x, int64_t, incx, cuDoubleComplex*, y, int64_t, incy);
DEF_FN(cublasStatus_t, cublasCopyEx, , cublasHandle_t, handle, int, n, const void*, x, cudaDataType, xType, int, incx, void*, y, cudaDataType, yType, int, incy);
DEF_FN(cublasStatus_t, cublasCopyEx_64, cublasHandle_t, handle, int64_t, n, const void*, x, cudaDataType, xType, int64_t, incx, void*, y, cudaDataType, yType, int64_t, incy);
DEF_FN(cublasStatus_t, cublasScopy_v2, cublasHandle_t, handle, int, n, const float*, x, int, incx, float*, y, int, incy);
DEF_FN(cublasStatus_t, cublasScopy_v2_64, cublasHandle_t, handle, int64_t, n, const float*, x, int64_t, incx, float*, y, int64_t, incy);
DEF_FN(cublasStatus_t, cublasDcopy_v2, cublasHandle_t, handle, int, n, const double*, x, int, incx, double*, y, int, incy);
DEF_FN(cublasStatus_t, cublasDcopy_v2_64, cublasHandle_t, handle, int64_t, n, const double*, x, int64_t, incx, double*, y, int64_t, incy);
DEF_FN(cublasStatus_t, cublasCcopy_v2, cublasHandle_t, handle, int, n, const cuComplex*, x, int, incx, cuComplex*, y, int, incy);
DEF_FN(cublasStatus_t, cublasCcopy_v2_64, cublasHandle_t, handle, int64_t, n, const cuComplex*, x, int64_t, incx, cuComplex*, y, int64_t, incy);
DEF_FN(cublasStatus_t, cublasZcopy_v2, cublasHandle_t, handle, int, n, const cuDoubleComplex*, x, int, incx, cuDoubleComplex*, y, int, incy);
DEF_FN(cublasStatus_t, cublasZcopy_v2_64, , cublasHandle_t, handle, int64_t, n, const cuDoubleComplex*, x, int64_t, incx, cuDoubleComplex*, y, int64_t, incy);
DEF_FN(cublasStatus_t, cublasSswap_v2, cublasHandle_t, handle, int, n, float*, x, int, incx, float*, y, int, incy);
DEF_FN(cublasStatus_t, cublasSswap_v2_64, cublasHandle_t, handle, int64_t, n, float*, x, int64_t, incx, float*, y, int64_t, incy);
DEF_FN(cublasStatus_t, cublasDswap_v2, cublasHandle_t, handle, int, n, double*, x, int, incx, double*, y, int, incy);
DEF_FN(cublasStatus_t, cublasDswap_v2_64, cublasHandle_t, handle, int64_t, n, double*, x, int64_t, incx, double*, y, int64_t, incy);
DEF_FN(cublasStatus_t, cublasCswap_v2, cublasHandle_t, handle, int, n, cuComplex*, x, int, incx, cuComplex*, y, int, incy);
DEF_FN(cublasStatus_t, cublasCswap_v2_64, cublasHandle_t, handle, int64_t, n, cuComplex*, x, int64_t, incx, cuComplex*, y, int64_t, incy);
DEF_FN(cublasStatus_t, cublasZswap_v2, cublasHandle_t, handle, int, n, cuDoubleComplex*, x, int, incx, cuDoubleComplex*, y, int, incy);
DEF_FN(cublasStatus_t, cublasZswap_v2_64, cublasHandle_t, handle, int64_t, n, cuDoubleComplex*, x, int64_t, incx, cuDoubleComplex*, y, int64_t, incy);
DEF_FN(cublasStatus_t, cublasSwapEx, , cublasHandle_t, handle, int, n, void*, x, cudaDataType, xType, int, incx, void*, y, cudaDataType, yType, int, incy);
DEF_FN(cublasStatus_t, cublasSwapEx_64, cublasHandle_t, handle, int64_t, n, void*, x, cudaDataType, xType, int64_t, incx, void*, y, cudaDataType, yType, int64_t, incy);
DEF_FN(cublasStatus_t, cublasIsamax_v2, cublasHandle_t, handle, int, n, const float*, x, int, incx, int*, result);
DEF_FN(cublasStatus_t, cublasIsamax_v2_64, cublasHandle_t, handle, int64_t, n, const float*, x, int64_t, incx, int64_t*, result);
DEF_FN(cublasStatus_t, cublasIdamax_v2, cublasHandle_t, handle, int, n, const double*, x, int, incx, int*, result);
DEF_FN(cublasStatus_t, cublasIdamax_v2_64, cublasHandle_t, handle, int64_t, n, const double*, x, int64_t, incx, int64_t*, result);
DEF_FN(cublasStatus_t, cublasIcamax_v2, cublasHandle_t, handle, int, n, const cuComplex*, x, int, incx, int*, result);
DEF_FN(cublasStatus_t, cublasIcamax_v2_64, cublasHandle_t, handle, int64_t, n, const cuComplex*, x, int64_t, incx, int64_t*, result);
DEF_FN(cublasStatus_t, cublasIzamax_v2, cublasHandle_t, handle, int, n, const cuDoubleComplex*, x, int, incx, int*, result);
DEF_FN(cublasStatus_t, cublasIzamax_v2_64, cublasHandle_t, handle, int64_t, n, const cuDoubleComplex*, x, int64_t, incx, int64_t*, result);
DEF_FN(cublasStatus_t, cublasIamaxEx, cublasHandle_t, handle, int, n, const void*, x, cudaDataType, xType, int, incx, int*, result);
DEF_FN(cublasStatus_t, cublasIamaxEx_64, cublasHandle_t, handle, int64_t, n, const void*, x, cudaDataType, xType, int64_t, incx, int64_t*, result);
DEF_FN(cublasStatus_t, cublasIsamin_v2, cublasHandle_t, handle, int, n, const float*, x, int, incx, int*, result);
DEF_FN(cublasStatus_t, cublasIsamin_v2_64, cublasHandle_t, handle, int64_t, n, const float*, x, int64_t, incx, int64_t*, result);
DEF_FN(cublasStatus_t, cublasIdamin_v2, cublasHandle_t, handle, int, n, const double*, x, int, incx, int*, result);
DEF_FN(cublasStatus_t, cublasIdamin_v2_64, cublasHandle_t, handle, int64_t, n, const double*, x, int64_t, incx, int64_t*, result);
DEF_FN(cublasStatus_t, cublasIcamin_v2, cublasHandle_t, handle, int, n, const cuComplex*, x, int, incx, int*, result);
DEF_FN(cublasStatus_t, cublasIcamin_v2_64, cublasHandle_t, handle, int64_t, n, const cuComplex*, x, int64_t, incx, int64_t*, result);
DEF_FN(cublasStatus_t, cublasIzamin_v2, cublasHandle_t, handle, int, n, const cuDoubleComplex*, x, int, incx, int*, result);
DEF_FN(cublasStatus_t, cublasIzamin_v2_64, cublasHandle_t, handle, int64_t, n, const cuDoubleComplex*, x, int64_t, incx, int64_t*, result);
DEF_FN(cublasStatus_t, cublasIaminEx, cublasHandle_t, handle, int, n, const void*, x, cudaDataType, xType, int, incx, int*, result);
DEF_FN(cublasStatus_t, cublasIaminEx_64, cublasHandle_t, handle, int64_t, n, const void*, x, cudaDataType, xType, int64_t, incx, int64_t*, result);
DEF_FN(cublasStatus_t, cublasAsumEx, cublasHandle_t, handle, int, n, const void*, x, cudaDataType, xType, int, incx, void*, result, cudaDataType, resultType, cudaDataType, executiontype);
DEF_FN(cublasStatus_t, cublasAsumEx_64, cublasHandle_t, handle, int64_t, n, const void*, x, cudaDataType, xType, int64_t, incx, void*, result, cudaDataType, resultType, cudaDataType, executiontype);
DEF_FN(cublasStatus_t, cublasSasum_v2, cublasHandle_t, handle, int, n, const float*, x, int, incx, float*, result);
DEF_FN(cublasStatus_t, cublasSasum_v2_64, cublasHandle_t, handle, int64_t, n, const float*, x, int64_t, incx, float*, result);
DEF_FN(cublasStatus_t, cublasDasum_v2, cublasHandle_t, handle, int, n, const double*, x, int, incx, double*, result);
DEF_FN(cublasStatus_t, cublasDasum_v2_64, cublasHandle_t, handle, int64_t, n, const double*, x, int64_t, incx, double*, result);
DEF_FN(cublasStatus_t, cublasScasum_v2, cublasHandle_t, handle, int, n, const cuComplex*, x, int, incx, float*, result);
DEF_FN(cublasStatus_t, cublasScasum_v2_64, cublasHandle_t, handle, int64_t, n, const cuComplex*, x, int64_t, incx, float*, result);
DEF_FN(cublasStatus_t, cublasDzasum_v2, cublasHandle_t, handle, int, n, const cuDoubleComplex*, x, int, incx, double*, result);
DEF_FN(cublasStatus_t, cublasDzasum_v2_64, cublasHandle_t, handle, int64_t, n, const cuDoubleComplex*, x, int64_t, incx, double*, result);
DEF_FN(cublasStatus_t, cublasSrot_v2, cublasHandle_t, handle, int, n, float*, x, int, incx, float*, y, int, incy, const float*, c, const float*, s);
DEF_FN(cublasStatus_t, cublasSrot_v2_64, , cublasHandle_t, handle, int64_t, n, float*, x, int64_t, incx, float*, y, int64_t, incy, const float*, c, const float*, s);
DEF_FN(cublasStatus_t, cublasDrot_v2, cublasHandle_t, handle, int, n, double*, x, int, incx, double*, y, int, incy, const double*, c, const double*, s);
DEF_FN(cublasStatus_t, cublasDrot_v2_64, cublasHandle_t, handle, int64_t, n, double*, x, int64_t, incx, double*, y, int64_t, incy, const double*, c, const double*, s);
DEF_FN(cublasStatus_t, cublasCrot_v2, , cublasHandle_t, handle, int, n, cuComplex*, x, int, incx, cuComplex*, y, int, incy, const float*, c, const cuComplex*, s);
DEF_FN(cublasStatus_t, cublasCrot_v2_64, cublasHandle_t, handle, int64_t, n, cuComplex*, x, int64_t, incx, cuComplex*, y, int64_t, incy, const float*, c, const cuComplex*, s);
DEF_FN(cublasStatus_t, cublasCsrot_v2, , cublasHandle_t, handle, int, n, cuComplex*, x, int, incx, cuComplex*, y, int, incy, const float*, c, const float*, s);
DEF_FN(cublasStatus_t, cublasCsrot_v2_64, cublasHandle_t, handle, int64_t, n, cuComplex*, x, int64_t, incx, cuComplex*, y, int64_t, incy, const float*, c, const float*, s);
DEF_FN(cublasStatus_t, cublasZrot_v2, cublasHandle_t, handle, int, n, cuDoubleComplex*, x, int, incx, cuDoubleComplex*, y, int, incy, const double*, c, const cuDoubleComplex*, s);
DEF_FN(cublasStatus_t, cublasZrot_v2_64, cublasHandle_t, handle, int64_t, n, cuDoubleComplex*, x, int64_t, incx, cuDoubleComplex*, y, int64_t, incy, const double*, c, const cuDoubleComplex*, s);
DEF_FN(cublasStatus_t, cublasZdrot_v2, cublasHandle_t, handle, int, n, cuDoubleComplex*, x, int, incx, cuDoubleComplex*, y, int, incy, const double*, c, const double*, s);
DEF_FN(cublasStatus_t, cublasZdrot_v2_64, cublasHandle_t, handle, int64_t, n, cuDoubleComplex*, x, int64_t, incx, cuDoubleComplex*, y, int64_t, incy, const double*, c, const double*, s);
DEF_FN(cublasStatus_t, cublasRotEx, cublasHandle_t, handle, int, n, void*, x, cudaDataType, xType, int, incx, void*, y, cudaDataType, yType, int, incy, const void*, c, const void*, s, cudaDataType, csType, cudaDataType, executiontype);
DEF_FN(cublasStatus_t, cublasRotEx_64, cublasHandle_t, handle, int64_t, n, void*, x, cudaDataType, xType, int64_t, incx, void*, y, cudaDataType, yType, int64_t, incy, const void*, c, const void*, s, cudaDataType, csType, cudaDataType, executiontype);
DEF_FN(cublasStatus_t, cublasSrotg_v2, cublasHandle_t, handle, float*, a, float*, b, float*, c, float*, s);
DEF_FN(cublasStatus_t, cublasDrotg_v2, cublasHandle_t, handle, double*, a, double*, b, double*, c, double*, s);
DEF_FN(cublasStatus_t, cublasCrotg_v2, cublasHandle_t, handle, cuComplex*, a, cuComplex*, b, float*, c, cuComplex*, s);
DEF_FN(cublasStatus_t, cublasZrotg_v2, cublasHandle_t, handle, cuDoubleComplex*, a, cuDoubleComplex*, b, double*, c, cuDoubleComplex*, s);
DEF_FN(cublasStatus_t, cublasRotgEx, cublasHandle_t, handle, void*, a, void*, b, cudaDataType, abType, void*, c, void*, s, cudaDataType, csType, cudaDataType, executiontype);
DEF_FN(cublasStatus_t, cublasSrotm_v2, cublasHandle_t, handle, int, n, float*, x, int, incx, float*, y, int, incy, const float*, param);
DEF_FN(cublasStatus_t, cublasSrotm_v2_64, cublasHandle_t, handle, int64_t, n, float*, x, int64_t, incx, float*, y, int64_t, incy, const float*, param);
DEF_FN(cublasStatus_t, cublasDrotm_v2, cublasHandle_t, handle, int, n, double*, x, int, incx, double*, y, int, incy, const double*, param);
DEF_FN(cublasStatus_t, cublasDrotm_v2_64, , cublasHandle_t, handle, int64_t, n, double*, x, int64_t, incx, double*, y, int64_t, incy, const double*, param);
DEF_FN(cublasStatus_t, cublasRotmEx, cublasHandle_t, handle, int, n, void*, x, cudaDataType, xType, int, incx, void*, y, cudaDataType, yType, int, incy, const void*, param, cudaDataType, paramType, cudaDataType, executiontype);
DEF_FN(cublasStatus_t, cublasRotmEx_64, cublasHandle_t, handle, int64_t, n, void*, x, cudaDataType, xType, int64_t, incx, void*, y, cudaDataType, yType, int64_t, incy, const void*, param, cudaDataType, paramType, cudaDataType, executiontype);
DEF_FN(cublasStatus_t, cublasSrotmg_v2, cublasHandle_t, handle, float*, d1, float*, d2, float*, x1, const float*, y1, float*, param);
DEF_FN(cublasStatus_t, cublasDrotmg_v2, cublasHandle_t, handle, double*, d1, double*, d2, double*, x1, const double*, y1, double*, param);
DEF_FN(cublasStatus_t, cublasRotmgEx, cublasHandle_t, handle, void*, d1, cudaDataType, d1Type, void*, d2, cudaDataType, d2Type, void*, x1, cudaDataType, x1Type, const void*, y1, cudaDataType, y1Type, void*, param, cudaDataType, paramType, cudaDataType, executiontype);

cublasStatus_t cublasSgemv_v2(cublasHandle_t handle,
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

DEF_FN(cublasStatus_t, cublasSgemv_v2_64, cublasHandle_t, handle, cublasOperation_t, trans, int64_t, m, int64_t, n, const float*, alpha, const float*, A, int64_t, lda, const float*, x, int64_t, incx, const float*, beta, float*, y, int64_t, incy);

cublasStatus_t cublasDgemv_v2(cublasHandle_t handle,
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
DEF_FN(cublasStatus_t, cublasDgemv_v2_64, cublasHandle_t, handle, cublasOperation_t, trans, int64_t, m, int64_t, n, const double*, alpha, const double*, A, int64_t, lda, const double*, x, int64_t, incx, const double*, beta, double*, y, int64_t, incy);
DEF_FN(cublasStatus_t, cublasCgemv_v2, cublasHandle_t, handle, cublasOperation_t, trans, int, m, int, n, const cuComplex*, alpha, const cuComplex*, A, int, lda, const cuComplex*, x, int, incx, const cuComplex*, beta, cuComplex*, y, int, incy);
DEF_FN(cublasStatus_t, cublasCgemv_v2_64, cublasHandle_t, handle, cublasOperation_t, trans, int64_t, m, int64_t, n, const cuComplex*, alpha, const cuComplex*, A, int64_t, lda, const cuComplex*, x, int64_t, incx, const cuComplex*, beta, cuComplex*, y, int64_t, incy);
DEF_FN(cublasStatus_t, cublasZgemv_v2, cublasHandle_t, handle, cublasOperation_t, trans, int, m, int, n, const cuDoubleComplex*, alpha, const cuDoubleComplex*, A, int, lda, const cuDoubleComplex*, x, int, incx, const cuDoubleComplex*, beta, cuDoubleComplex*, y, int, incy);
DEF_FN(cublasStatus_t, cublasZgemv_v2_64, cublasHandle_t, handle, cublasOperation_t, trans, int64_t, m, int64_t, n, const cuDoubleComplex*, alpha, const cuDoubleComplex*, A, int64_t, lda, const cuDoubleComplex*, x, int64_t, incx, const cuDoubleComplex*, beta, cuDoubleComplex*, y, int64_t, incy);
DEF_FN(cublasStatus_t, cublasSgbmv_v2, cublasHandle_t, handle, cublasOperation_t, trans, int, m, int, n, int, kl, int, ku, const float*, alpha, const float*, A, int, lda, const float*, x, int, incx, const float*, beta, float*, y, int, incy);
DEF_FN(cublasStatus_t, cublasSgbmv_v2_64, cublasHandle_t, handle, cublasOperation_t, trans, int64_t, m, int64_t, n, int64_t, kl, int64_t, ku, const float*, alpha, const float*, A, int64_t, lda, const float*, x, int64_t, incx, const float*, beta, float*, y, int64_t, incy);
DEF_FN(cublasStatus_t, cublasDgbmv_v2, cublasHandle_t, handle, cublasOperation_t, trans, int, m, int, n, int, kl, int, ku, const double*, alpha, const double*, A, int, lda, const double*, x, int, incx, const double*, beta, double*, y, int, incy);
DEF_FN(cublasStatus_t, cublasDgbmv_v2_64, cublasHandle_t, handle, cublasOperation_t, trans, int64_t, m, int64_t, n, int64_t, kl, int64_t, ku, const double*, alpha, const double*, A, int64_t, lda, const double*, x, int64_t, incx, const double*, beta, double*, y, int64_t, incy);
DEF_FN(cublasStatus_t, cublasCgbmv_v2, cublasHandle_t, handle, cublasOperation_t, trans, int, m, int, n, int, kl, int, ku, const cuComplex*, alpha, const cuComplex*, A, int, lda, const cuComplex*, x, int, incx, const cuComplex*, beta, cuComplex*, y, int, incy);
DEF_FN(cublasStatus_t, cublasCgbmv_v2_64, cublasHandle_t, handle, cublasOperation_t, trans, int64_t, m, int64_t, n, int64_t, kl, int64_t, ku, const cuComplex*, alpha, const cuComplex*, A, int64_t, lda, const cuComplex*, x, int64_t, incx, const cuComplex*, beta, cuComplex*, y, int64_t, incy);
DEF_FN(cublasStatus_t, cublasZgbmv_v2, cublasHandle_t, handle, cublasOperation_t, trans, int, m, int, n, int, kl, int, ku, const cuDoubleComplex*, alpha, const cuDoubleComplex*, A, int, lda, const cuDoubleComplex*, x, int, incx, const cuDoubleComplex*, beta, cuDoubleComplex*, y, int, incy);
DEF_FN(cublasStatus_t, cublasZgbmv_v2_64, cublasHandle_t, handle, cublasOperation_t, trans, int64_t, m, int64_t, n, int64_t, kl, int64_t, ku, const cuDoubleComplex*, alpha, const cuDoubleComplex*, A, int64_t, lda, const cuDoubleComplex*, x, int64_t, incx, const cuDoubleComplex*, beta, cuDoubleComplex*, y, int64_t, incy);
DEF_FN(cublasStatus_t, cublasStrmv_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int, n, const float*, A, int, lda, float*, x, int, incx);
DEF_FN(cublasStatus_t, cublasStrmv_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int64_t, n, const float*, A, int64_t, lda, float*, x, int64_t, incx);
DEF_FN(cublasStatus_t, cublasDtrmv_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int, n, const double*, A, int, lda, double*, x, int, incx);
DEF_FN(cublasStatus_t, cublasDtrmv_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int64_t, n, const double*, A, int64_t, lda, double*, x, int64_t, incx);
DEF_FN(cublasStatus_t, cublasCtrmv_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int, n, const cuComplex*, A, int, lda, cuComplex*, x, int, incx);
DEF_FN(cublasStatus_t, cublasCtrmv_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int64_t, n, const cuComplex*, A, int64_t, lda, cuComplex*, x, int64_t, incx);
DEF_FN(cublasStatus_t, cublasZtrmv_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int, n, const cuDoubleComplex*, A, int, lda, cuDoubleComplex*, x, int, incx);
DEF_FN(cublasStatus_t, cublasZtrmv_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int64_t, n, const cuDoubleComplex*, A, int64_t, lda, cuDoubleComplex*, x, int64_t, incx);
DEF_FN(cublasStatus_t, cublasStbmv_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int, n, int, k, const float*, A, int, lda, float*, x, int, incx);
DEF_FN(cublasStatus_t, cublasStbmv_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int64_t, n, int64_t, k, const float*, A, int64_t, lda, float*, x, int64_t, incx);
DEF_FN(cublasStatus_t, cublasDtbmv_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int, n, int, k, const double*, A, int, lda, double*, x, int, incx);
DEF_FN(cublasStatus_t, cublasDtbmv_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int64_t, n, int64_t, k, const double*, A, int64_t, lda, double*, x, int64_t, incx);
DEF_FN(cublasStatus_t, cublasCtbmv_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int, n, int, k, const cuComplex*, A, int, lda, cuComplex*, x, int, incx);
DEF_FN(cublasStatus_t, cublasCtbmv_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int64_t, n, int64_t, k, const cuComplex*, A, int64_t, lda, cuComplex*, x, int64_t, incx);
DEF_FN(cublasStatus_t, cublasZtbmv_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int, n, int, k, const cuDoubleComplex*, A, int, lda, cuDoubleComplex*, x, int, incx);
DEF_FN(cublasStatus_t, cublasZtbmv_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int64_t, n, int64_t, k, const cuDoubleComplex*, A, int64_t, lda, cuDoubleComplex*, x, int64_t, incx);
DEF_FN(cublasStatus_t, cublasStpmv_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int, n, const float*, AP, float*, x, int, incx);
DEF_FN(cublasStatus_t, cublasStpmv_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int64_t, n, const float*, AP, float*, x, int64_t, incx);
DEF_FN(cublasStatus_t, cublasDtpmv_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int, n, const double*, AP, double*, x, int, incx);
DEF_FN(cublasStatus_t, cublasDtpmv_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int64_t, n, const double*, AP, double*, x, int64_t, incx);
DEF_FN(cublasStatus_t, cublasCtpmv_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int, n, const cuComplex*, AP, cuComplex*, x, int, incx);
DEF_FN(cublasStatus_t, cublasCtpmv_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int64_t, n, const cuComplex*, AP, cuComplex*, x, int64_t, incx);
DEF_FN(cublasStatus_t, cublasZtpmv_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int, n, const cuDoubleComplex*, AP, cuDoubleComplex*, x, int, incx);
DEF_FN(cublasStatus_t, cublasZtpmv_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int64_t, n, const cuDoubleComplex*, AP, cuDoubleComplex*, x, int64_t, incx);
DEF_FN(cublasStatus_t, cublasStrsv_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int, n, const float*, A, int, lda, float*, x, int, incx);
DEF_FN(cublasStatus_t, cublasStrsv_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int64_t, n, const float*, A, int64_t, lda, float*, x, int64_t, incx);
DEF_FN(cublasStatus_t, cublasDtrsv_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int, n, const double*, A, int, lda, double*, x, int, incx);
DEF_FN(cublasStatus_t, cublasDtrsv_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int64_t, n, const double*, A, int64_t, lda, double*, x, int64_t, incx);
DEF_FN(cublasStatus_t, cublasCtrsv_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int, n, const cuComplex*, A, int, lda, cuComplex*, x, int, incx);
DEF_FN(cublasStatus_t, cublasCtrsv_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int64_t, n, const cuComplex*, A, int64_t, lda, cuComplex*, x, int64_t, incx);
DEF_FN(cublasStatus_t, cublasZtrsv_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int, n, const cuDoubleComplex*, A, int, lda, cuDoubleComplex*, x, int, incx);
DEF_FN(cublasStatus_t, cublasZtrsv_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int64_t, n, const cuDoubleComplex*, A, int64_t, lda, cuDoubleComplex*, x, int64_t, incx);
DEF_FN(cublasStatus_t, cublasStpsv_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int, n, const float*, AP, float*, x, int, incx);
DEF_FN(cublasStatus_t, cublasStpsv_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int64_t, n, const float*, AP, float*, x, int64_t, incx);
DEF_FN(cublasStatus_t, cublasDtpsv_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int, n, const double*, AP, double*, x, int, incx);
DEF_FN(cublasStatus_t, cublasDtpsv_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int64_t, n, const double*, AP, double*, x, int64_t, incx);
DEF_FN(cublasStatus_t, cublasCtpsv_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int, n, const cuComplex*, AP, cuComplex*, x, int, incx);
DEF_FN(cublasStatus_t, cublasCtpsv_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int64_t, n, const cuComplex*, AP, cuComplex*, x, int64_t, incx);
DEF_FN(cublasStatus_t, cublasZtpsv_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int, n, const cuDoubleComplex*, AP, cuDoubleComplex*, x, int, incx);
DEF_FN(cublasStatus_t, cublasZtpsv_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int64_t, n, const cuDoubleComplex*, AP, cuDoubleComplex*, x, int64_t, incx);
DEF_FN(cublasStatus_t, cublasStbsv_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int, n, int, k, const float*, A, int, lda, float*, x, int, incx);
DEF_FN(cublasStatus_t, cublasStbsv_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int64_t, n, int64_t, k, const float*, A, int64_t, lda, float*, x, int64_t, incx);
DEF_FN(cublasStatus_t, cublasDtbsv_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int, n, int, k, const double*, A, int, lda, double*, x, int, incx);
DEF_FN(cublasStatus_t, cublasDtbsv_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int64_t, n, int64_t, k, const double*, A, int64_t, lda, double*, x, int64_t, incx);
DEF_FN(cublasStatus_t, cublasCtbsv_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int, n, int, k, const cuComplex*, A, int, lda, cuComplex*, x, int, incx);
DEF_FN(cublasStatus_t, cublasCtbsv_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int64_t, n, int64_t, k, const cuComplex*, A, int64_t, lda, cuComplex*, x, int64_t, incx);
DEF_FN(cublasStatus_t, cublasZtbsv_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int, n, int, k, const cuDoubleComplex*, A, int, lda, cuDoubleComplex*, x, int, incx);
DEF_FN(cublasStatus_t, cublasZtbsv_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int64_t, n, int64_t, k, const cuDoubleComplex*, A, int64_t, lda, cuDoubleComplex*, x, int64_t, incx);
DEF_FN(cublasStatus_t, cublasSsymv_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, int, n, const float*, alpha, const float*, A, int, lda, const float*, x, int, incx, const float*, beta, float*, y, int, incy);
DEF_FN(cublasStatus_t, cublasSsymv_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, int64_t, n, const float*, alpha, const float*, A, int64_t, lda, const float*, x, int64_t, incx, const float*, beta, float*, y, int64_t, incy);
DEF_FN(cublasStatus_t, cublasDsymv_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, int, n, const double*, alpha, const double*, A, int, lda, const double*, x, int, incx, const double*, beta, double*, y, int, incy);
DEF_FN(cublasStatus_t, cublasDsymv_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, int64_t, n, const double*, alpha, const double*, A, int64_t, lda, const double*, x, int64_t, incx, const double*, beta, double*, y, int64_t, incy);
DEF_FN(cublasStatus_t, cublasCsymv_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, int, n, const cuComplex*, alpha, const cuComplex*, A, int, lda, const cuComplex*, x, int, incx, const cuComplex*, beta, cuComplex*, y, int, incy);
DEF_FN(cublasStatus_t, cublasCsymv_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, int64_t, n, const cuComplex*, alpha, const cuComplex*, A, int64_t, lda, const cuComplex*, x, int64_t, incx, const cuComplex*, beta, cuComplex*, y, int64_t, incy);
DEF_FN(cublasStatus_t, cublasZsymv_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, int, n, const cuDoubleComplex*, alpha, const cuDoubleComplex*, A, int, lda, const cuDoubleComplex*, x, int, incx, const cuDoubleComplex*, beta, cuDoubleComplex*, y, int, incy);
DEF_FN(cublasStatus_t, cublasZsymv_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, int64_t, n, const cuDoubleComplex*, alpha, const cuDoubleComplex*, A, int64_t, lda, const cuDoubleComplex*, x, int64_t, incx, const cuDoubleComplex*, beta, cuDoubleComplex*, y, int64_t, incy);
DEF_FN(cublasStatus_t, cublasChemv_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, int, n, const cuComplex*, alpha, const cuComplex*, A, int, lda, const cuComplex*, x, int, incx, const cuComplex*, beta, cuComplex*, y, int, incy);
DEF_FN(cublasStatus_t, cublasChemv_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, int64_t, n, const cuComplex*, alpha, const cuComplex*, A, int64_t, lda, const cuComplex*, x, int64_t, incx, const cuComplex*, beta, cuComplex*, y, int64_t, incy);
DEF_FN(cublasStatus_t, cublasZhemv_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, int, n, const cuDoubleComplex*, alpha, const cuDoubleComplex*, A, int, lda, const cuDoubleComplex*, x, int, incx, const cuDoubleComplex*, beta, cuDoubleComplex*, y, int, incy);
DEF_FN(cublasStatus_t, cublasZhemv_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, int64_t, n, const cuDoubleComplex*, alpha, const cuDoubleComplex*, A, int64_t, lda, const cuDoubleComplex*, x, int64_t, incx, const cuDoubleComplex*, beta, cuDoubleComplex*, y, int64_t, incy);
DEF_FN(cublasStatus_t, cublasSsbmv_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, int, n, int, k, const float*, alpha, const float*, A, int, lda, const float*, x, int, incx, const float*, beta, float*, y, int, incy);
DEF_FN(cublasStatus_t, cublasSsbmv_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, int64_t, n, int64_t, k, const float*, alpha, const float*, A, int64_t, lda, const float*, x, int64_t, incx, const float*, beta, float*, y, int64_t, incy);
DEF_FN(cublasStatus_t, cublasDsbmv_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, int, n, int, k, const double*, alpha, const double*, A, int, lda, const double*, x, int, incx, const double*, beta, double*, y, int, incy);
DEF_FN(cublasStatus_t, cublasDsbmv_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, int64_t, n, int64_t, k, const double*, alpha, const double*, A, int64_t, lda, const double*, x, int64_t, incx, const double*, beta, double*, y, int64_t, incy);
DEF_FN(cublasStatus_t, cublasChbmv_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, int, n, int, k, const cuComplex*, alpha, const cuComplex*, A, int, lda, const cuComplex*, x, int, incx, const cuComplex*, beta, cuComplex*, y, int, incy);
DEF_FN(cublasStatus_t, cublasChbmv_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, int64_t, n, int64_t, k, const cuComplex*, alpha, const cuComplex*, A, int64_t, lda, const cuComplex*, x, int64_t, incx, const cuComplex*, beta, cuComplex*, y, int64_t, incy);
DEF_FN(cublasStatus_t, cublasZhbmv_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, int, n, int, k, const cuDoubleComplex*, alpha, const cuDoubleComplex*, A, int, lda, const cuDoubleComplex*, x, int, incx, const cuDoubleComplex*, beta, cuDoubleComplex*, y, int, incy);
DEF_FN(cublasStatus_t, cublasZhbmv_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, int64_t, n, int64_t, k, const cuDoubleComplex*, alpha, const cuDoubleComplex*, A, int64_t, lda, const cuDoubleComplex*, x, int64_t, incx, const cuDoubleComplex*, beta, cuDoubleComplex*, y, int64_t, incy);
DEF_FN(cublasStatus_t, cublasSspmv_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, int, n, const float*, alpha, const float*, AP, const float*, x, int, incx, const float*, beta, float*, y, int, incy);
DEF_FN(cublasStatus_t, cublasSspmv_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, int64_t, n, const float*, alpha, const float*, AP, const float*, x, int64_t, incx, const float*, beta, float*, y, int64_t, incy);
DEF_FN(cublasStatus_t, cublasDspmv_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, int, n, const double*, alpha, const double*, AP, const double*, x, int, incx, const double*, beta, double*, y, int, incy);
DEF_FN(cublasStatus_t, cublasDspmv_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, int64_t, n, const double*, alpha, const double*, AP, const double*, x, int64_t, incx, const double*, beta, double*, y, int64_t, incy);
DEF_FN(cublasStatus_t, cublasChpmv_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, int, n, const cuComplex*, alpha, const cuComplex*, AP, const cuComplex*, x, int, incx, const cuComplex*, beta, cuComplex*, y, int, incy);
DEF_FN(cublasStatus_t, cublasChpmv_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, int64_t, n, const cuComplex*, alpha, const cuComplex*, AP, const cuComplex*, x, int64_t, incx, const cuComplex*, beta, cuComplex*, y, int64_t, incy);
DEF_FN(cublasStatus_t, cublasZhpmv_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, int, n, const cuDoubleComplex*, alpha, const cuDoubleComplex*, AP, const cuDoubleComplex*, x, int, incx, const cuDoubleComplex*, beta, cuDoubleComplex*, y, int, incy);
DEF_FN(cublasStatus_t, cublasZhpmv_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, int64_t, n, const cuDoubleComplex*, alpha, const cuDoubleComplex*, AP, const cuDoubleComplex*, x, int64_t, incx, const cuDoubleComplex*, beta, cuDoubleComplex*, y, int64_t, incy);
DEF_FN(cublasStatus_t, cublasSger_v2, cublasHandle_t, handle, int, m, int, n, const float*, alpha, const float*, x, int, incx, const float*, y, int, incy, float*, A, int, lda);
DEF_FN(cublasStatus_t, cublasSger_v2_64, cublasHandle_t, handle, int64_t, m, int64_t, n, const float*, alpha, const float*, x, int64_t, incx, const float*, y, int64_t, incy, float*, A, int64_t, lda);
DEF_FN(cublasStatus_t, cublasDger_v2, cublasHandle_t, handle, int, m, int, n, const double*, alpha, const double*, x, int, incx, const double*, y, int, incy, double*, A, int, lda);
DEF_FN(cublasStatus_t, cublasDger_v2_64, cublasHandle_t, handle, int64_t, m, int64_t, n, const double*, alpha, const double*, x, int64_t, incx, const double*, y, int64_t, incy, double*, A, int64_t, lda);
DEF_FN(cublasStatus_t, cublasCgeru_v2, cublasHandle_t, handle, int, m, int, n, const cuComplex*, alpha, const cuComplex*, x, int, incx, const cuComplex*, y, int, incy, cuComplex*, A, int, lda);
DEF_FN(cublasStatus_t, cublasCgeru_v2_64, cublasHandle_t, handle, int64_t, m, int64_t, n, const cuComplex*, alpha, const cuComplex*, x, int64_t, incx, const cuComplex*, y, int64_t, incy, cuComplex*, A, int64_t, lda);
DEF_FN(cublasStatus_t, cublasCgerc_v2, cublasHandle_t, handle, int, m, int, n, const cuComplex*, alpha, const cuComplex*, x, int, incx, const cuComplex*, y, int, incy, cuComplex*, A, int, lda);
DEF_FN(cublasStatus_t, cublasCgerc_v2_64, cublasHandle_t, handle, int64_t, m, int64_t, n, const cuComplex*, alpha, const cuComplex*, x, int64_t, incx, const cuComplex*, y, int64_t, incy, cuComplex*, A, int64_t, lda);
DEF_FN(cublasStatus_t, cublasZgeru_v2, cublasHandle_t, handle, int, m, int, n, const cuDoubleComplex*, alpha, const cuDoubleComplex*, x, int, incx, const cuDoubleComplex*, y, int, incy, cuDoubleComplex*, A, int, lda);
DEF_FN(cublasStatus_t, cublasZgeru_v2_64, cublasHandle_t, handle, int64_t, m, int64_t, n, const cuDoubleComplex*, alpha, const cuDoubleComplex*, x, int64_t, incx, const cuDoubleComplex*, y, int64_t, incy, cuDoubleComplex*, A, int64_t, lda);
DEF_FN(cublasStatus_t, cublasZgerc_v2, cublasHandle_t, handle, int, m, int, n, const cuDoubleComplex*, alpha, const cuDoubleComplex*, x, int, incx, const cuDoubleComplex*, y, int, incy, cuDoubleComplex*, A, int, lda);
DEF_FN(cublasStatus_t, cublasZgerc_v2_64, cublasHandle_t, handle, int64_t, m, int64_t, n, const cuDoubleComplex*, alpha, const cuDoubleComplex*, x, int64_t, incx, const cuDoubleComplex*, y, int64_t, incy, cuDoubleComplex*, A, int64_t, lda);
DEF_FN(cublasStatus_t, cublasSsyr_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, int, n, const float*, alpha, const float*, x, int, incx, float*, A, int, lda);
DEF_FN(cublasStatus_t, cublasSsyr_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, int64_t, n, const float*, alpha, const float*, x, int64_t, incx, float*, A, int64_t, lda);
DEF_FN(cublasStatus_t, cublasDsyr_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, int, n, const double*, alpha, const double*, x, int, incx, double*, A, int, lda);
DEF_FN(cublasStatus_t, cublasDsyr_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, int64_t, n, const double*, alpha, const double*, x, int64_t, incx, double*, A, int64_t, lda);
DEF_FN(cublasStatus_t, cublasCsyr_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, int, n, const cuComplex*, alpha, const cuComplex*, x, int, incx, cuComplex*, A, int, lda);
DEF_FN(cublasStatus_t, cublasCsyr_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, int64_t, n, const cuComplex*, alpha, const cuComplex*, x, int64_t, incx, cuComplex*, A, int64_t, lda);
DEF_FN(cublasStatus_t, cublasZsyr_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, int, n, const cuDoubleComplex*, alpha, const cuDoubleComplex*, x, int, incx, cuDoubleComplex*, A, int, lda);
DEF_FN(cublasStatus_t, cublasZsyr_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, int64_t, n, const cuDoubleComplex*, alpha, const cuDoubleComplex*, x, int64_t, incx, cuDoubleComplex*, A, int64_t, lda);
DEF_FN(cublasStatus_t, cublasCher_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, int, n, const float*, alpha, const cuComplex*, x, int, incx, cuComplex*, A, int, lda);
DEF_FN(cublasStatus_t, cublasCher_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, int64_t, n, const float*, alpha, const cuComplex*, x, int64_t, incx, cuComplex*, A, int64_t, lda);
DEF_FN(cublasStatus_t, cublasZher_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, int, n, const double*, alpha, const cuDoubleComplex*, x, int, incx, cuDoubleComplex*, A, int, lda);
DEF_FN(cublasStatus_t, cublasZher_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, int64_t, n, const double*, alpha, const cuDoubleComplex*, x, int64_t, incx, cuDoubleComplex*, A, int64_t, lda);
DEF_FN(cublasStatus_t, cublasSspr_v2, , cublasHandle_t, handle, cublasFillMode_t, uplo, int, n, const float*, alpha, const float*, x, int, incx, float*, AP);
DEF_FN(cublasStatus_t, cublasSspr_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, int64_t, n, const float*, alpha, const float*, x, int64_t, incx, float*, AP);
DEF_FN(cublasStatus_t, cublasDspr_v2, , cublasHandle_t, handle, cublasFillMode_t, uplo, int, n, const double*, alpha, const double*, x, int, incx, double*, AP);
DEF_FN(cublasStatus_t, cublasDspr_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, int64_t, n, const double*, alpha, const double*, x, int64_t, incx, double*, AP);
DEF_FN(cublasStatus_t, cublasChpr_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, int, n, const float*, alpha, const cuComplex*, x, int, incx, cuComplex*, AP);
DEF_FN(cublasStatus_t, cublasChpr_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, int64_t, n, const float*, alpha, const cuComplex*, x, int64_t, incx, cuComplex*, AP);
DEF_FN(cublasStatus_t, cublasZhpr_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, int, n, const double*, alpha, const cuDoubleComplex*, x, int, incx, cuDoubleComplex*, AP);
DEF_FN(cublasStatus_t, cublasZhpr_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, int64_t, n, const double*, alpha, const cuDoubleComplex*, x, int64_t, incx, cuDoubleComplex*, AP);
DEF_FN(cublasStatus_t, cublasSsyr2_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, int, n, const float*, alpha, const float*, x, int, incx, const float*, y, int, incy, float*, A, int, lda);
DEF_FN(cublasStatus_t, cublasSsyr2_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, int64_t, n, const float*, alpha, const float*, x, int64_t, incx, const float*, y, int64_t, incy, float*, A, int64_t, lda);
DEF_FN(cublasStatus_t, cublasDsyr2_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, int, n, const double*, alpha, const double*, x, int, incx, const double*, y, int, incy, double*, A, int, lda);
DEF_FN(cublasStatus_t, cublasDsyr2_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, int64_t, n, const double*, alpha, const double*, x, int64_t, incx, const double*, y, int64_t, incy, double*, A, int64_t, lda);
DEF_FN(cublasStatus_t, cublasCsyr2_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, int, n, const cuComplex*, alpha, const cuComplex*, x, int, incx, const cuComplex*, y, int, incy, cuComplex*, A, int, lda);
DEF_FN(cublasStatus_t, cublasCsyr2_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, int64_t, n, const cuComplex*, alpha, const cuComplex*, x, int64_t, incx, const cuComplex*, y, int64_t, incy, cuComplex*, A, int64_t, lda);
DEF_FN(cublasStatus_t, cublasZsyr2_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, int, n, const cuDoubleComplex*, alpha, const cuDoubleComplex*, x, int, incx, const cuDoubleComplex*, y, int, incy, cuDoubleComplex*, A, int, lda);
DEF_FN(cublasStatus_t, cublasZsyr2_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, int64_t, n, const cuDoubleComplex*, alpha, const cuDoubleComplex*, x, int64_t, incx, const cuDoubleComplex*, y, int64_t, incy, cuDoubleComplex*, A, int64_t, lda);
DEF_FN(cublasStatus_t, cublasCher2_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, int, n, const cuComplex*, alpha, const cuComplex*, x, int, incx, const cuComplex*, y, int, incy, cuComplex*, A, int, lda);
DEF_FN(cublasStatus_t, cublasCher2_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, int64_t, n, const cuComplex*, alpha, const cuComplex*, x, int64_t, incx, const cuComplex*, y, int64_t, incy, cuComplex*, A, int64_t, lda);
DEF_FN(cublasStatus_t, cublasZher2_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, int, n, const cuDoubleComplex*, alpha, const cuDoubleComplex*, x, int, incx, const cuDoubleComplex*, y, int, incy, cuDoubleComplex*, A, int, lda);
DEF_FN(cublasStatus_t, cublasZher2_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, int64_t, n, const cuDoubleComplex*, alpha, const cuDoubleComplex*, x, int64_t, incx, const cuDoubleComplex*, y, int64_t, incy, cuDoubleComplex*, A, int64_t, lda);
DEF_FN(cublasStatus_t, cublasSspr2_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, int, n, const float*, alpha, const float*, x, int, incx, const float*, y, int, incy, float*, AP);
DEF_FN(cublasStatus_t, cublasSspr2_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, int64_t, n, const float*, alpha, const float*, x, int64_t, incx, const float*, y, int64_t, incy, float*, AP);
DEF_FN(cublasStatus_t, cublasDspr2_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, int, n, const double*, alpha, const double*, x, int, incx, const double*, y, int, incy, double*, AP);
DEF_FN(cublasStatus_t, cublasDspr2_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, int64_t, n, const double*, alpha, const double*, x, int64_t, incx, const double*, y, int64_t, incy, double*, AP);
DEF_FN(cublasStatus_t, cublasChpr2_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, int, n, const cuComplex*, alpha, const cuComplex*, x, int, incx, const cuComplex*, y, int, incy, cuComplex*, AP);
DEF_FN(cublasStatus_t, cublasChpr2_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, int64_t, n, const cuComplex*, alpha, const cuComplex*, x, int64_t, incx, const cuComplex*, y, int64_t, incy, cuComplex*, AP);
DEF_FN(cublasStatus_t, cublasZhpr2_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, int, n, const cuDoubleComplex*, alpha, const cuDoubleComplex*, x, int, incx, const cuDoubleComplex*, y, int, incy, cuDoubleComplex*, AP);
DEF_FN(cublasStatus_t, cublasZhpr2_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, int64_t, n, const cuDoubleComplex*, alpha, const cuDoubleComplex*, x, int64_t, incx, const cuDoubleComplex*, y, int64_t, incy, cuDoubleComplex*, AP);
DEF_FN(cublasStatus_t, cublasSgemvBatched, cublasHandle_t, handle, cublasOperation_t, trans, int, m, int, n, const float*, alpha, const float* const*,  Aarray, int, lda, const float* const*,  xarray, int, incx, const float*, beta, float* const*,  yarray, int, incy, int, batchCount);
DEF_FN(cublasStatus_t, cublasSgemvBatched_64, cublasHandle_t, handle, cublasOperation_t, trans, int64_t, m, int64_t, n, const float*, alpha, const float* const*,  Aarray, int64_t, lda, const float* const*,  xarray, int64_t, incx, const float*, beta, float* const*,  yarray, int64_t, incy, int64_t, batchCount);
DEF_FN(cublasStatus_t, cublasDgemvBatched, cublasHandle_t, handle, cublasOperation_t, trans, int, m, int, n, const double*, alpha, const double* const*,  Aarray, int, lda, const double* const*,  xarray, int, incx, const double*, beta, double* const*,  yarray, int, incy, int, batchCount);
DEF_FN(cublasStatus_t, cublasDgemvBatched_64, cublasHandle_t, handle, cublasOperation_t, trans, int64_t, m, int64_t, n, const double*, alpha, const double* const*,  Aarray, int64_t, lda, const double* const*,  xarray, int64_t, incx, const double*, beta, double* const*,  yarray, int64_t, incy, int64_t, batchCount);
DEF_FN(cublasStatus_t, cublasCgemvBatched, cublasHandle_t, handle, cublasOperation_t, trans, int, m, int, n, const cuComplex*, alpha, const cuComplex* const*,  Aarray, int, lda, const cuComplex* const*,  xarray, int, incx, const cuComplex*, beta, cuComplex* const*,  yarray, int, incy, int, batchCount);
DEF_FN(cublasStatus_t, cublasCgemvBatched_64, cublasHandle_t, handle, cublasOperation_t, trans, int64_t, m, int64_t, n, const cuComplex*, alpha, const cuComplex* const*,  Aarray, int64_t, lda, const cuComplex* const*,  xarray, int64_t, incx, const cuComplex*, beta, cuComplex* const*,  yarray, int64_t, incy, int64_t, batchCount);
DEF_FN(cublasStatus_t, cublasZgemvBatched, cublasHandle_t, handle, cublasOperation_t, trans, int, m, int, n, const cuDoubleComplex*, alpha, const cuDoubleComplex* const*,  Aarray, int, lda, const cuDoubleComplex* const*,  xarray, int, incx, const cuDoubleComplex*, beta, cuDoubleComplex* const*,  yarray, int, incy, int, batchCount);
DEF_FN(cublasStatus_t, cublasZgemvBatched_64, cublasHandle_t, handle, cublasOperation_t, trans, int64_t, m, int64_t, n, const cuDoubleComplex*, alpha, const cuDoubleComplex* const*,  Aarray, int64_t, lda, const cuDoubleComplex* const*,  xarray, int64_t, incx, const cuDoubleComplex*, beta, cuDoubleComplex* const*,  yarray, int64_t, incy, int64_t, batchCount);
DEF_FN(cublasStatus_t, cublasSgemvStridedBatched, cublasHandle_t, handle, cublasOperation_t, trans, int, m, int, n, const float*, alpha, const float*, A, int, lda, long long int, strideA, const float*, x, int, incx, long long int, stridex, const float*, beta, float*, y, int, incy, long long int, stridey, int, batchCount);
DEF_FN(cublasStatus_t, cublasSgemvStridedBatched_64, cublasHandle_t, handle, cublasOperation_t, trans, int64_t, m, int64_t, n, const float*, alpha, const float*, A, int64_t, lda, long long int, strideA, const float*, x, int64_t, incx, long long int, stridex, const float*, beta, float*, y, int64_t, incy, long long int, stridey, int64_t, batchCount);
DEF_FN(cublasStatus_t, cublasDgemvStridedBatched, cublasHandle_t, handle, cublasOperation_t, trans, int, m, int, n, const double*, alpha, const double*, A, int, lda, long long int, strideA, const double*, x, int, incx, long long int, stridex, const double*, beta, double*, y, int, incy, long long int, stridey, int, batchCount);
DEF_FN(cublasStatus_t, cublasDgemvStridedBatched_64, cublasHandle_t, handle, cublasOperation_t, trans, int64_t, m, int64_t, n, const double*, alpha, const double*, A, int64_t, lda, long long int, strideA, const double*, x, int64_t, incx, long long int, stridex, const double*, beta, double*, y, int64_t, incy, long long int, stridey, int64_t, batchCount);
DEF_FN(cublasStatus_t, cublasCgemvStridedBatched, cublasHandle_t, handle, cublasOperation_t, trans, int, m, int, n, const cuComplex*, alpha, const cuComplex*, A, int, lda, long long int, strideA, const cuComplex*, x, int, incx, long long int, stridex, const cuComplex*, beta, cuComplex*, y, int, incy, long long int, stridey, int, batchCount);
DEF_FN(cublasStatus_t, cublasCgemvStridedBatched_64, cublasHandle_t, handle, cublasOperation_t, trans, int64_t, m, int64_t, n, const cuComplex*, alpha, const cuComplex*, A, int64_t, lda, long long int, strideA, const cuComplex*, x, int64_t, incx, long long int, stridex, const cuComplex*, beta, cuComplex*, y, int64_t, incy, long long int, stridey, int64_t, batchCount);
DEF_FN(cublasStatus_t, cublasZgemvStridedBatched, cublasHandle_t, handle, cublasOperation_t, trans, int, m, int, n, const cuDoubleComplex*, alpha, const cuDoubleComplex*, A, int, lda, long long int, strideA, const cuDoubleComplex*, x, int, incx, long long int, stridex, const cuDoubleComplex*, beta, cuDoubleComplex*, y, int, incy, long long int, stridey, int, batchCount);
DEF_FN(cublasStatus_t, cublasZgemvStridedBatched_64, cublasHandle_t, handle, cublasOperation_t, trans, int64_t, m, int64_t, n, const cuDoubleComplex*, alpha, const cuDoubleComplex*, A, int64_t, lda, long long int, strideA, const cuDoubleComplex*, x, int64_t, incx, long long int, stridex, const cuDoubleComplex*, beta, cuDoubleComplex*, y, int64_t, incy, long long int, stridey, int64_t, batchCount);

cublasStatus_t cublasSgemm_v2(cublasHandle_t handle,
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
DEF_FN(cublasStatus_t, cublasSgemm_v2_64, cublasHandle_t, handle, cublasOperation_t, transa, cublasOperation_t, transb, int64_t, m, int64_t, n, int64_t, k, const float*, alpha, const float*, A, int64_t, lda, const float*, B, int64_t, ldb, const float*, beta, float*, C, int64_t, ldc);

cublasStatus_t cublasDgemm_v2(cublasHandle_t handle,
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

DEF_FN(cublasStatus_t, cublasDgemm_v2_64, cublasHandle_t, handle, cublasOperation_t, transa, cublasOperation_t, transb, int64_t, m, int64_t, n, int64_t, k, const double*, alpha, const double*, A, int64_t, lda, const double*, B, int64_t, ldb, const double*, beta, double*, C, int64_t, ldc);
DEF_FN(cublasStatus_t, cublasCgemm_v2, cublasHandle_t, handle, cublasOperation_t, transa, cublasOperation_t, transb, int, m, int, n, int, k, const cuComplex*, alpha, const cuComplex*, A, int, lda, const cuComplex*, B, int, ldb, const cuComplex*, beta, cuComplex*, C, int, ldc);
DEF_FN(cublasStatus_t, cublasCgemm_v2_64, cublasHandle_t, handle, cublasOperation_t, transa, cublasOperation_t, transb, int64_t, m, int64_t, n, int64_t, k, const cuComplex*, alpha, const cuComplex*, A, int64_t, lda, const cuComplex*, B, int64_t, ldb, const cuComplex*, beta, cuComplex*, C, int64_t, ldc);
DEF_FN(cublasStatus_t, cublasCgemm3m, cublasHandle_t, handle, cublasOperation_t, transa, cublasOperation_t, transb, int, m, int, n, int, k, const cuComplex*, alpha, const cuComplex*, A, int, lda, const cuComplex*, B, int, ldb, const cuComplex*, beta, cuComplex*, C, int, ldc);
DEF_FN(cublasStatus_t, cublasCgemm3m_64, cublasHandle_t, handle, cublasOperation_t, transa, cublasOperation_t, transb, int64_t, m, int64_t, n, int64_t, k, const cuComplex*, alpha, const cuComplex*, A, int64_t, lda, const cuComplex*, B, int64_t, ldb, const cuComplex*, beta, cuComplex*, C, int64_t, ldc);
DEF_FN(cublasStatus_t, cublasCgemm3mEx, cublasHandle_t, handle, cublasOperation_t, transa, cublasOperation_t, transb, int, m, int, n, int, k, const cuComplex*, alpha, const void*, A, cudaDataType, Atype, int, lda, const void*, B, cudaDataType, Btype, int, ldb, const cuComplex*, beta, void*, C, cudaDataType, Ctype, int, ldc);
DEF_FN(cublasStatus_t, cublasCgemm3mEx_64, cublasHandle_t, handle, cublasOperation_t, transa, cublasOperation_t, transb, int64_t, m, int64_t, n, int64_t, k, const cuComplex*, alpha, const void*, A, cudaDataType, Atype, int64_t, lda, const void*, B, cudaDataType, Btype, int64_t, ldb, const cuComplex*, beta, void*, C, cudaDataType, Ctype, int64_t, ldc);
DEF_FN(cublasStatus_t, cublasZgemm_v2, cublasHandle_t, handle, cublasOperation_t, transa, cublasOperation_t, transb, int, m, int, n, int, k, const cuDoubleComplex*, alpha, const cuDoubleComplex*, A, int, lda, const cuDoubleComplex*, B, int, ldb, const cuDoubleComplex*, beta, cuDoubleComplex*, C, int, ldc);
DEF_FN(cublasStatus_t, cublasZgemm_v2_64, cublasHandle_t, handle, cublasOperation_t, transa, cublasOperation_t, transb, int64_t, m, int64_t, n, int64_t, k, const cuDoubleComplex*, alpha, const cuDoubleComplex*, A, int64_t, lda, const cuDoubleComplex*, B, int64_t, ldb, const cuDoubleComplex*, beta, cuDoubleComplex*, C, int64_t, ldc);
DEF_FN(cublasStatus_t, cublasZgemm3m, cublasHandle_t, handle, cublasOperation_t, transa, cublasOperation_t, transb, int, m, int, n, int, k, const cuDoubleComplex*, alpha, const cuDoubleComplex*, A, int, lda, const cuDoubleComplex*, B, int, ldb, const cuDoubleComplex*, beta, cuDoubleComplex*, C, int, ldc);
DEF_FN(cublasStatus_t, cublasZgemm3m_64, cublasHandle_t, handle, cublasOperation_t, transa, cublasOperation_t, transb, int64_t, m, int64_t, n, int64_t, k, const cuDoubleComplex*, alpha, const cuDoubleComplex*, A, int64_t, lda, const cuDoubleComplex*, B, int64_t, ldb, const cuDoubleComplex*, beta, cuDoubleComplex*, C, int64_t, ldc);

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


DEF_FN(cublasStatus_t, cublasSgemmEx_64, cublasHandle_t, handle, cublasOperation_t, transa, cublasOperation_t, transb, int64_t, m, int64_t, n, int64_t, k, const float*, alpha, const void*, A, cudaDataType, Atype, int64_t, lda, const void*, B, cudaDataType, Btype, int64_t, ldb, const float*, beta, void*, C, cudaDataType, Ctype, int64_t, ldc);
DEF_FN(cublasStatus_t, cublasGemmEx_64, cublasHandle_t, handle, cublasOperation_t, transa, cublasOperation_t, transb, int64_t, m, int64_t, n, int64_t, k, const void*, alpha, const void*, A, cudaDataType, Atype, int64_t, lda, const void*, B, cudaDataType, Btype, int64_t, ldb, const void*, beta, void*, C, cudaDataType, Ctype, int64_t, ldc, cublasComputeType_t, computeType, cublasGemmAlgo_t, algo);
DEF_FN(cublasStatus_t, cublasCgemmEx, cublasHandle_t, handle, cublasOperation_t, transa, cublasOperation_t, transb, int, m, int, n, int, k, const cuComplex*, alpha, const void*, A, cudaDataType, Atype, int, lda, const void*, B, cudaDataType, Btype, int, ldb, const cuComplex*, beta, void*, C, cudaDataType, Ctype, int, ldc);
DEF_FN(cublasStatus_t, cublasCgemmEx_64, cublasHandle_t, handle, cublasOperation_t, transa, cublasOperation_t, transb, int64_t, m, int64_t, n, int64_t, k, const cuComplex*, alpha, const void*, A, cudaDataType, Atype, int64_t, lda, const void*, B, cudaDataType, Btype, int64_t, ldb, const cuComplex*, beta, void*, C, cudaDataType, Ctype, int64_t, ldc);
DEF_FN(cublasStatus_t, cublasSsyrk_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, int, n, int, k, const float*, alpha, const float*, A, int, lda, const float*, beta, float*, C, int, ldc);
DEF_FN(cublasStatus_t, cublasSsyrk_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, int64_t, n, int64_t, k, const float*, alpha, const float*, A, int64_t, lda, const float*, beta, float*, C, int64_t, ldc);
DEF_FN(cublasStatus_t, cublasDsyrk_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, int, n, int, k, const double*, alpha, const double*, A, int, lda, const double*, beta, double*, C, int, ldc);
DEF_FN(cublasStatus_t, cublasDsyrk_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, int64_t, n, int64_t, k, const double*, alpha, const double*, A, int64_t, lda, const double*, beta, double*, C, int64_t, ldc);
DEF_FN(cublasStatus_t, cublasCsyrk_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, int, n, int, k, const cuComplex*, alpha, const cuComplex*, A, int, lda, const cuComplex*, beta, cuComplex*, C, int, ldc);
DEF_FN(cublasStatus_t, cublasCsyrk_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, int64_t, n, int64_t, k, const cuComplex*, alpha, const cuComplex*, A, int64_t, lda, const cuComplex*, beta, cuComplex*, C, int64_t, ldc);
DEF_FN(cublasStatus_t, cublasZsyrk_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, int, n, int, k, const cuDoubleComplex*, alpha, const cuDoubleComplex*, A, int, lda, const cuDoubleComplex*, beta, cuDoubleComplex*, C, int, ldc);
DEF_FN(cublasStatus_t, cublasZsyrk_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, int64_t, n, int64_t, k, const cuDoubleComplex*, alpha, const cuDoubleComplex*, A, int64_t, lda, const cuDoubleComplex*, beta, cuDoubleComplex*, C, int64_t, ldc);
DEF_FN(cublasStatus_t, cublasCsyrkEx, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, int, n, int, k, const cuComplex*, alpha, const void*, A, cudaDataType, Atype, int, lda, const cuComplex*, beta, void*, C, cudaDataType, Ctype, int, ldc);
DEF_FN(cublasStatus_t, cublasCsyrkEx_64, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, int64_t, n, int64_t, k, const cuComplex*, alpha, const void*, A, cudaDataType, Atype, int64_t, lda, const cuComplex*, beta, void*, C, cudaDataType, Ctype, int64_t, ldc);
DEF_FN(cublasStatus_t, cublasCsyrk3mEx, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, int, n, int, k, const cuComplex*, alpha, const void*, A, cudaDataType, Atype, int, lda, const cuComplex*, beta, void*, C, cudaDataType, Ctype, int, ldc);
DEF_FN(cublasStatus_t, cublasCsyrk3mEx_64, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, int64_t, n, int64_t, k, const cuComplex*, alpha, const void*, A, cudaDataType, Atype, int64_t, lda, const cuComplex*, beta, void*, C, cudaDataType, Ctype, int64_t, ldc);
DEF_FN(cublasStatus_t, cublasCherk_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, int, n, int, k, const float*, alpha, const cuComplex*, A, int, lda, const float*, beta, cuComplex*, C, int, ldc);
DEF_FN(cublasStatus_t, cublasCherk_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, int64_t, n, int64_t, k, const float*, alpha, const cuComplex*, A, int64_t, lda, const float*, beta, cuComplex*, C, int64_t, ldc);
DEF_FN(cublasStatus_t, cublasZherk_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, int, n, int, k, const double*, alpha, const cuDoubleComplex*, A, int, lda, const double*, beta, cuDoubleComplex*, C, int, ldc);
DEF_FN(cublasStatus_t, cublasZherk_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, int64_t, n, int64_t, k, const double*, alpha, const cuDoubleComplex*, A, int64_t, lda, const double*, beta, cuDoubleComplex*, C, int64_t, ldc);
DEF_FN(cublasStatus_t, cublasCherkEx, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, int, n, int, k, const float*, alpha, const void*, A, cudaDataType, Atype, int, lda, const float*, beta, void*, C, cudaDataType, Ctype, int, ldc);
DEF_FN(cublasStatus_t, cublasCherkEx_64, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, int64_t, n, int64_t, k, const float*, alpha, const void*, A, cudaDataType, Atype, int64_t, lda, const float*, beta, void*, C, cudaDataType, Ctype, int64_t, ldc);
DEF_FN(cublasStatus_t, cublasCherk3mEx, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, int, n, int, k, const float*, alpha, const void*, A, cudaDataType, Atype, int, lda, const float*, beta, void*, C, cudaDataType, Ctype, int, ldc);
DEF_FN(cublasStatus_t, cublasCherk3mEx_64, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, int64_t, n, int64_t, k, const float*, alpha, const void*, A, cudaDataType, Atype, int64_t, lda, const float*, beta, void*, C, cudaDataType, Ctype, int64_t, ldc);
DEF_FN(cublasStatus_t, cublasSsyr2k_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, int, n, int, k, const float*, alpha, const float*, A, int, lda, const float*, B, int, ldb, const float*, beta, float*, C, int, ldc);
DEF_FN(cublasStatus_t, cublasSsyr2k_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, int64_t, n, int64_t, k, const float*, alpha, const float*, A, int64_t, lda, const float*, B, int64_t, ldb, const float*, beta, float*, C, int64_t, ldc);
DEF_FN(cublasStatus_t, cublasDsyr2k_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, int, n, int, k, const double*, alpha, const double*, A, int, lda, const double*, B, int, ldb, const double*, beta, double*, C, int, ldc);
DEF_FN(cublasStatus_t, cublasDsyr2k_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, int64_t, n, int64_t, k, const double*, alpha, const double*, A, int64_t, lda, const double*, B, int64_t, ldb, const double*, beta, double*, C, int64_t, ldc);
DEF_FN(cublasStatus_t, cublasCsyr2k_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, int, n, int, k, const cuComplex*, alpha, const cuComplex*, A, int, lda, const cuComplex*, B, int, ldb, const cuComplex*, beta, cuComplex*, C, int, ldc);
DEF_FN(cublasStatus_t, cublasCsyr2k_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, int64_t, n, int64_t, k, const cuComplex*, alpha, const cuComplex*, A, int64_t, lda, const cuComplex*, B, int64_t, ldb, const cuComplex*, beta, cuComplex*, C, int64_t, ldc);
DEF_FN(cublasStatus_t, cublasZsyr2k_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, int, n, int, k, const cuDoubleComplex*, alpha, const cuDoubleComplex*, A, int, lda, const cuDoubleComplex*, B, int, ldb, const cuDoubleComplex*, beta, cuDoubleComplex*, C, int, ldc);
DEF_FN(cublasStatus_t, cublasZsyr2k_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, int64_t, n, int64_t, k, const cuDoubleComplex*, alpha, const cuDoubleComplex*, A, int64_t, lda, const cuDoubleComplex*, B, int64_t, ldb, const cuDoubleComplex*, beta, cuDoubleComplex*, C, int64_t, ldc);
DEF_FN(cublasStatus_t, cublasCher2k_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, int, n, int, k, const cuComplex*, alpha, const cuComplex*, A, int, lda, const cuComplex*, B, int, ldb, const float*, beta, cuComplex*, C, int, ldc);
DEF_FN(cublasStatus_t, cublasCher2k_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, int64_t, n, int64_t, k, const cuComplex*, alpha, const cuComplex*, A, int64_t, lda, const cuComplex*, B, int64_t, ldb, const float*, beta, cuComplex*, C, int64_t, ldc);
DEF_FN(cublasStatus_t, cublasZher2k_v2, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, int, n, int, k, const cuDoubleComplex*, alpha, const cuDoubleComplex*, A, int, lda, const cuDoubleComplex*, B, int, ldb, const double*, beta, cuDoubleComplex*, C, int, ldc);
DEF_FN(cublasStatus_t, cublasZher2k_v2_64, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, int64_t, n, int64_t, k, const cuDoubleComplex*, alpha, const cuDoubleComplex*, A, int64_t, lda, const cuDoubleComplex*, B, int64_t, ldb, const double*, beta, cuDoubleComplex*, C, int64_t, ldc);
DEF_FN(cublasStatus_t, cublasSsyrkx, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, int, n, int, k, const float*, alpha, const float*, A, int, lda, const float*, B, int, ldb, const float*, beta, float*, C, int, ldc);
DEF_FN(cublasStatus_t, cublasSsyrkx_64, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, int64_t, n, int64_t, k, const float*, alpha, const float*, A, int64_t, lda, const float*, B, int64_t, ldb, const float*, beta, float*, C, int64_t, ldc);
DEF_FN(cublasStatus_t, cublasDsyrkx, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, int, n, int, k, const double*, alpha, const double*, A, int, lda, const double*, B, int, ldb, const double*, beta, double*, C, int, ldc);
DEF_FN(cublasStatus_t, cublasDsyrkx_64, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, int64_t, n, int64_t, k, const double*, alpha, const double*, A, int64_t, lda, const double*, B, int64_t, ldb, const double*, beta, double*, C, int64_t, ldc);
DEF_FN(cublasStatus_t, cublasCsyrkx, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, int, n, int, k, const cuComplex*, alpha, const cuComplex*, A, int, lda, const cuComplex*, B, int, ldb, const cuComplex*, beta, cuComplex*, C, int, ldc);
DEF_FN(cublasStatus_t, cublasCsyrkx_64, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, int64_t, n, int64_t, k, const cuComplex*, alpha, const cuComplex*, A, int64_t, lda, const cuComplex*, B, int64_t, ldb, const cuComplex*, beta, cuComplex*, C, int64_t, ldc);
DEF_FN(cublasStatus_t, cublasZsyrkx, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, int, n, int, k, const cuDoubleComplex*, alpha, const cuDoubleComplex*, A, int, lda, const cuDoubleComplex*, B, int, ldb, const cuDoubleComplex*, beta, cuDoubleComplex*, C, int, ldc);
DEF_FN(cublasStatus_t, cublasZsyrkx_64, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, int64_t, n, int64_t, k, const cuDoubleComplex*, alpha, const cuDoubleComplex*, A, int64_t, lda, const cuDoubleComplex*, B, int64_t, ldb, const cuDoubleComplex*, beta, cuDoubleComplex*, C, int64_t, ldc);
DEF_FN(cublasStatus_t, cublasCherkx, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, int, n, int, k, const cuComplex*, alpha, const cuComplex*, A, int, lda, const cuComplex*, B, int, ldb, const float*, beta, cuComplex*, C, int, ldc);
DEF_FN(cublasStatus_t, cublasCherkx_64, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, int64_t, n, int64_t, k, const cuComplex*, alpha, const cuComplex*, A, int64_t, lda, const cuComplex*, B, int64_t, ldb, const float*, beta, cuComplex*, C, int64_t, ldc);
DEF_FN(cublasStatus_t, cublasZherkx, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, int, n, int, k, const cuDoubleComplex*, alpha, const cuDoubleComplex*, A, int, lda, const cuDoubleComplex*, B, int, ldb, const double*, beta, cuDoubleComplex*, C, int, ldc);
DEF_FN(cublasStatus_t, cublasZherkx_64, cublasHandle_t, handle, cublasFillMode_t, uplo, cublasOperation_t, trans, int64_t, n, int64_t, k, const cuDoubleComplex*, alpha, const cuDoubleComplex*, A, int64_t, lda, const cuDoubleComplex*, B, int64_t, ldb, const double*, beta, cuDoubleComplex*, C, int64_t, ldc);
DEF_FN(cublasStatus_t, cublasSsymm_v2, cublasHandle_t, handle, cublasSideMode_t, side, cublasFillMode_t, uplo, int, m, int, n, const float*, alpha, const float*, A, int, lda, const float*, B, int, ldb, const float*, beta, float*, C, int, ldc);
DEF_FN(cublasStatus_t, cublasSsymm_v2_64, cublasHandle_t, handle, cublasSideMode_t, side, cublasFillMode_t, uplo, int64_t, m, int64_t, n, const float*, alpha, const float*, A, int64_t, lda, const float*, B, int64_t, ldb, const float*, beta, float*, C, int64_t, ldc);
DEF_FN(cublasStatus_t, cublasDsymm_v2, cublasHandle_t, handle, cublasSideMode_t, side, cublasFillMode_t, uplo, int, m, int, n, const double*, alpha, const double*, A, int, lda, const double*, B, int, ldb, const double*, beta, double*, C, int, ldc);
DEF_FN(cublasStatus_t, cublasDsymm_v2_64, cublasHandle_t, handle, cublasSideMode_t, side, cublasFillMode_t, uplo, int64_t, m, int64_t, n, const double*, alpha, const double*, A, int64_t, lda, const double*, B, int64_t, ldb, const double*, beta, double*, C, int64_t, ldc);
DEF_FN(cublasStatus_t, cublasCsymm_v2, cublasHandle_t, handle, cublasSideMode_t, side, cublasFillMode_t, uplo, int, m, int, n, const cuComplex*, alpha, const cuComplex*, A, int, lda, const cuComplex*, B, int, ldb, const cuComplex*, beta, cuComplex*, C, int, ldc);
DEF_FN(cublasStatus_t, cublasCsymm_v2_64, cublasHandle_t, handle, cublasSideMode_t, side, cublasFillMode_t, uplo, int64_t, m, int64_t, n, const cuComplex*, alpha, const cuComplex*, A, int64_t, lda, const cuComplex*, B, int64_t, ldb, const cuComplex*, beta, cuComplex*, C, int64_t, ldc);
DEF_FN(cublasStatus_t, cublasZsymm_v2, cublasHandle_t, handle, cublasSideMode_t, side, cublasFillMode_t, uplo, int, m, int, n, const cuDoubleComplex*, alpha, const cuDoubleComplex*, A, int, lda, const cuDoubleComplex*, B, int, ldb, const cuDoubleComplex*, beta, cuDoubleComplex*, C, int, ldc);
DEF_FN(cublasStatus_t, cublasZsymm_v2_64, cublasHandle_t, handle, cublasSideMode_t, side, cublasFillMode_t, uplo, int64_t, m, int64_t, n, const cuDoubleComplex*, alpha, const cuDoubleComplex*, A, int64_t, lda, const cuDoubleComplex*, B, int64_t, ldb, const cuDoubleComplex*, beta, cuDoubleComplex*, C, int64_t, ldc);
DEF_FN(cublasStatus_t, cublasChemm_v2, cublasHandle_t, handle, cublasSideMode_t, side, cublasFillMode_t, uplo, int, m, int, n, const cuComplex*, alpha, const cuComplex*, A, int, lda, const cuComplex*, B, int, ldb, const cuComplex*, beta, cuComplex*, C, int, ldc);
DEF_FN(cublasStatus_t, cublasChemm_v2_64, cublasHandle_t, handle, cublasSideMode_t, side, cublasFillMode_t, uplo, int64_t, m, int64_t, n, const cuComplex*, alpha, const cuComplex*, A, int64_t, lda, const cuComplex*, B, int64_t, ldb, const cuComplex*, beta, cuComplex*, C, int64_t, ldc);
DEF_FN(cublasStatus_t, cublasZhemm_v2, cublasHandle_t, handle, cublasSideMode_t, side, cublasFillMode_t, uplo, int, m, int, n, const cuDoubleComplex*, alpha, const cuDoubleComplex*, A, int, lda, const cuDoubleComplex*, B, int, ldb, const cuDoubleComplex*, beta, cuDoubleComplex*, C, int, ldc);
DEF_FN(cublasStatus_t, cublasZhemm_v2_64, cublasHandle_t, handle, cublasSideMode_t, side, cublasFillMode_t, uplo, int64_t, m, int64_t, n, const cuDoubleComplex*, alpha, const cuDoubleComplex*, A, int64_t, lda, const cuDoubleComplex*, B, int64_t, ldb, const cuDoubleComplex*, beta, cuDoubleComplex*, C, int64_t, ldc);
DEF_FN(cublasStatus_t, cublasStrsm_v2, cublasHandle_t, handle, cublasSideMode_t, side, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int, m, int, n, const float*, alpha, const float*, A, int, lda, float*, B, int, ldb);
DEF_FN(cublasStatus_t, cublasStrsm_v2_64, cublasHandle_t, handle, cublasSideMode_t, side, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int64_t, m, int64_t, n, const float*, alpha, const float*, A, int64_t, lda, float*, B, int64_t, ldb);
DEF_FN(cublasStatus_t, cublasDtrsm_v2, cublasHandle_t, handle, cublasSideMode_t, side, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int, m, int, n, const double*, alpha, const double*, A, int, lda, double*, B, int, ldb);
DEF_FN(cublasStatus_t, cublasDtrsm_v2_64, cublasHandle_t, handle, cublasSideMode_t, side, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int64_t, m, int64_t, n, const double*, alpha, const double*, A, int64_t, lda, double*, B, int64_t, ldb);
DEF_FN(cublasStatus_t, cublasCtrsm_v2, cublasHandle_t, handle, cublasSideMode_t, side, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int, m, int, n, const cuComplex*, alpha, const cuComplex*, A, int, lda, cuComplex*, B, int, ldb);
DEF_FN(cublasStatus_t, cublasCtrsm_v2_64, cublasHandle_t, handle, cublasSideMode_t, side, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int64_t, m, int64_t, n, const cuComplex*, alpha, const cuComplex*, A, int64_t, lda, cuComplex*, B, int64_t, ldb);
DEF_FN(cublasStatus_t, cublasZtrsm_v2, cublasHandle_t, handle, cublasSideMode_t, side, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int, m, int, n, const cuDoubleComplex*, alpha, const cuDoubleComplex*, A, int, lda, cuDoubleComplex*, B, int, ldb);
DEF_FN(cublasStatus_t, cublasZtrsm_v2_64, cublasHandle_t, handle, cublasSideMode_t, side, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int64_t, m, int64_t, n, const cuDoubleComplex*, alpha, const cuDoubleComplex*, A, int64_t, lda, cuDoubleComplex*, B, int64_t, ldb);
DEF_FN(cublasStatus_t, cublasStrmm_v2, cublasHandle_t, handle, cublasSideMode_t, side, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int, m, int, n, const float*, alpha, const float*, A, int, lda, const float*, B, int, ldb, float*, C, int, ldc);
DEF_FN(cublasStatus_t, cublasStrmm_v2_64, cublasHandle_t, handle, cublasSideMode_t, side, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int64_t, m, int64_t, n, const float*, alpha, const float*, A, int64_t, lda, const float*, B, int64_t, ldb, float*, C, int64_t, ldc);
DEF_FN(cublasStatus_t, cublasDtrmm_v2, cublasHandle_t, handle, cublasSideMode_t, side, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int, m, int, n, const double*, alpha, const double*, A, int, lda, const double*, B, int, ldb, double*, C, int, ldc);
DEF_FN(cublasStatus_t, cublasDtrmm_v2_64, cublasHandle_t, handle, cublasSideMode_t, side, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int64_t, m, int64_t, n, const double*, alpha, const double*, A, int64_t, lda, const double*, B, int64_t, ldb, double*, C, int64_t, ldc);
DEF_FN(cublasStatus_t, cublasCtrmm_v2, cublasHandle_t, handle, cublasSideMode_t, side, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int, m, int, n, const cuComplex*, alpha, const cuComplex*, A, int, lda, const cuComplex*, B, int, ldb, cuComplex*, C, int, ldc);
DEF_FN(cublasStatus_t, cublasCtrmm_v2_64, cublasHandle_t, handle, cublasSideMode_t, side, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int64_t, m, int64_t, n, const cuComplex*, alpha, const cuComplex*, A, int64_t, lda, const cuComplex*, B, int64_t, ldb, cuComplex*, C, int64_t, ldc);
DEF_FN(cublasStatus_t, cublasZtrmm_v2, cublasHandle_t, handle, cublasSideMode_t, side, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int, m, int, n, const cuDoubleComplex*, alpha, const cuDoubleComplex*, A, int, lda, const cuDoubleComplex*, B, int, ldb, cuDoubleComplex*, C, int, ldc);
DEF_FN(cublasStatus_t, cublasZtrmm_v2_64, cublasHandle_t, handle, cublasSideMode_t, side, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int64_t, m, int64_t, n, const cuDoubleComplex*, alpha, const cuDoubleComplex*, A, int64_t, lda, const cuDoubleComplex*, B, int64_t, ldb, cuDoubleComplex*, C, int64_t, ldc);
DEF_FN(cublasStatus_t, cublasSgemmBatched, cublasHandle_t, handle, cublasOperation_t, transa, cublasOperation_t, transb, int, m, int, n, int, k, const float*, alpha, const float* const*,  Aarray, int, lda, const float* const*,  Barray, int, ldb, const float*, beta, float* const*,  Carray, int, ldc, int, batchCount);
DEF_FN(cublasStatus_t, cublasSgemmBatched_64, cublasHandle_t, handle, cublasOperation_t, transa, cublasOperation_t, transb, int64_t, m, int64_t, n, int64_t, k, const float*, alpha, const float* const*,  Aarray, int64_t, lda, const float* const*,  Barray, int64_t, ldb, const float*, beta, float* const*,  Carray, int64_t, ldc, int64_t, batchCount);
DEF_FN(cublasStatus_t, cublasDgemmBatched, cublasHandle_t, handle, cublasOperation_t, transa, cublasOperation_t, transb, int, m, int, n, int, k, const double*, alpha, const double* const*,  Aarray, int, lda, const double* const*,  Barray, int, ldb, const double*, beta, double* const*,  Carray, int, ldc, int, batchCount);
DEF_FN(cublasStatus_t, cublasDgemmBatched_64, cublasHandle_t, handle, cublasOperation_t, transa, cublasOperation_t, transb, int64_t, m, int64_t, n, int64_t, k, const double*, alpha, const double* const*,  Aarray, int64_t, lda, const double* const*,  Barray, int64_t, ldb, const double*, beta, double* const*,  Carray, int64_t, ldc, int64_t, batchCount);
DEF_FN(cublasStatus_t, cublasCgemmBatched, cublasHandle_t, handle, cublasOperation_t, transa, cublasOperation_t, transb, int, m, int, n, int, k, const cuComplex*, alpha, const cuComplex* const*,  Aarray, int, lda, const cuComplex* const*,  Barray, int, ldb, const cuComplex*, beta, cuComplex* const*,  Carray, int, ldc, int, batchCount);
DEF_FN(cublasStatus_t, cublasCgemmBatched_64, cublasHandle_t, handle, cublasOperation_t, transa, cublasOperation_t, transb, int64_t, m, int64_t, n, int64_t, k, const cuComplex*, alpha, const cuComplex* const*,  Aarray, int64_t, lda, const cuComplex* const*,  Barray, int64_t, ldb, const cuComplex*, beta, cuComplex* const*,  Carray, int64_t, ldc, int64_t, batchCount);
DEF_FN(cublasStatus_t, cublasCgemm3mBatched, cublasHandle_t, handle, cublasOperation_t, transa, cublasOperation_t, transb, int, m, int, n, int, k, const cuComplex*, alpha, const cuComplex* const*,  Aarray, int, lda, const cuComplex* const*,  Barray, int, ldb, const cuComplex*, beta, cuComplex* const*,  Carray, int, ldc, int, batchCount);
DEF_FN(cublasStatus_t, cublasCgemm3mBatched_64, cublasHandle_t, handle, cublasOperation_t, transa, cublasOperation_t, transb, int64_t, m, int64_t, n, int64_t, k, const cuComplex*, alpha, const cuComplex* const*,  Aarray, int64_t, lda, const cuComplex* const*,  Barray, int64_t, ldb, const cuComplex*, beta, cuComplex* const*,  Carray, int64_t, ldc, int64_t, batchCount);
DEF_FN(cublasStatus_t, cublasZgemmBatched, cublasHandle_t, handle, cublasOperation_t, transa, cublasOperation_t, transb, int, m, int, n, int, k, const cuDoubleComplex*, alpha, const cuDoubleComplex* const*,  Aarray, int, lda, const cuDoubleComplex* const*,  Barray, int, ldb, const cuDoubleComplex*, beta, cuDoubleComplex* const*,  Carray, int, ldc, int, batchCount);
DEF_FN(cublasStatus_t, cublasZgemmBatched_64, cublasHandle_t, handle, cublasOperation_t, transa, cublasOperation_t, transb, int64_t, m, int64_t, n, int64_t, k, const cuDoubleComplex*, alpha, const cuDoubleComplex* const*,  Aarray, int64_t, lda, const cuDoubleComplex* const*,  Barray, int64_t, ldb, const cuDoubleComplex*, beta, cuDoubleComplex* const*,  Carray, int64_t, ldc, int64_t, batchCount);
DEF_FN(cublasStatus_t, cublasSgemmStridedBatched_64, cublasHandle_t, handle, cublasOperation_t, transa, cublasOperation_t, transb, int64_t, m, int64_t, n, int64_t, k, const float*, alpha, const float*, A, int64_t, lda, long long int, strideA, const float*, B, int64_t, ldb, long long int, strideB, const float*, beta, float*, C, int64_t, ldc, long long int, strideC, int64_t, batchCount);
DEF_FN(cublasStatus_t, cublasDgemmStridedBatched, cublasHandle_t, handle, cublasOperation_t, transa, cublasOperation_t, transb, int, m, int, n, int, k, const double*, alpha, const double*, A, int, lda, long long int, strideA, const double*, B, int, ldb, long long int, strideB, const double*, beta, double*, C, int, ldc, long long int, strideC, int, batchCount);
DEF_FN(cublasStatus_t, cublasDgemmStridedBatched_64, cublasHandle_t, handle, cublasOperation_t, transa, cublasOperation_t, transb, int64_t, m, int64_t, n, int64_t, k, const double*, alpha, const double*, A, int64_t, lda, long long int, strideA, const double*, B, int64_t, ldb, long long int, strideB, const double*, beta, double*, C, int64_t, ldc, long long int, strideC, int64_t, batchCount);
DEF_FN(cublasStatus_t, cublasCgemmStridedBatched, cublasHandle_t, handle, cublasOperation_t, transa, cublasOperation_t, transb, int, m, int, n, int, k, const cuComplex*, alpha, const cuComplex*, A, int, lda, long long int, strideA, const cuComplex*, B, int, ldb, long long int, strideB, const cuComplex*, beta, cuComplex*, C, int, ldc, long long int, strideC, int, batchCount);
DEF_FN(cublasStatus_t, cublasCgemmStridedBatched_64, cublasHandle_t, handle, cublasOperation_t, transa, cublasOperation_t, transb, int64_t, m, int64_t, n, int64_t, k, const cuComplex*, alpha, const cuComplex*, A, int64_t, lda, long long int, strideA, const cuComplex*, B, int64_t, ldb, long long int, strideB, const cuComplex*, beta, cuComplex*, C, int64_t, ldc, long long int, strideC, int64_t, batchCount);
DEF_FN(cublasStatus_t, cublasCgemm3mStridedBatched, cublasHandle_t, handle, cublasOperation_t, transa, cublasOperation_t, transb, int, m, int, n, int, k, const cuComplex*, alpha, const cuComplex*, A, int, lda, long long int, strideA, const cuComplex*, B, int, ldb, long long int, strideB, const cuComplex*, beta, cuComplex*, C, int, ldc, long long int, strideC, int, batchCount);
DEF_FN(cublasStatus_t, cublasCgemm3mStridedBatched_64, cublasHandle_t, handle, cublasOperation_t, transa, cublasOperation_t, transb, int64_t, m, int64_t, n, int64_t, k, const cuComplex*, alpha, const cuComplex*, A, int64_t, lda, long long int, strideA, const cuComplex*, B, int64_t, ldb, long long int, strideB, const cuComplex*, beta, cuComplex*, C, int64_t, ldc, long long int, strideC, int64_t, batchCount);
DEF_FN(cublasStatus_t, cublasZgemmStridedBatched, cublasHandle_t, handle, cublasOperation_t, transa, cublasOperation_t, transb, int, m, int, n, int, k, const cuDoubleComplex*, alpha, const cuDoubleComplex*, A, int, lda, long long int, strideA, const cuDoubleComplex*, B, int, ldb, long long int, strideB, const cuDoubleComplex*, beta, cuDoubleComplex*, C, int, ldc, long long int, strideC, int, batchCount);
DEF_FN(cublasStatus_t, cublasZgemmStridedBatched_64, cublasHandle_t, handle, cublasOperation_t, transa, cublasOperation_t, transb, int64_t, m, int64_t, n, int64_t, k, const cuDoubleComplex*, alpha, const cuDoubleComplex*, A, int64_t, lda, long long int, strideA, const cuDoubleComplex*, B, int64_t, ldb, long long int, strideB, const cuDoubleComplex*, beta, cuDoubleComplex*, C, int64_t, ldc, long long int, strideC, int64_t, batchCount);
DEF_FN(cublasStatus_t, cublasGemmBatchedEx, cublasHandle_t, handle, cublasOperation_t, transa, cublasOperation_t, transb, int, m, int, n, int, k, const void*, alpha, const void* const*,  Aarray, cudaDataType, Atype, int, lda, const void* const*,  Barray, cudaDataType, Btype, int, ldb, const void*, beta, void* const*,  Carray, cudaDataType, Ctype, int, ldc, int, batchCount, cublasComputeType_t, computeType, cublasGemmAlgo_t, algo);
DEF_FN(cublasStatus_t, cublasGemmBatchedEx_64, cublasHandle_t, handle, cublasOperation_t, transa, cublasOperation_t, transb, int64_t, m, int64_t, n, int64_t, k, const void*, alpha, const void* const*,  Aarray, cudaDataType, Atype, int64_t, lda, const void* const*,  Barray, cudaDataType, Btype, int64_t, ldb, const void*, beta, void* const*,  Carray, cudaDataType, Ctype, int64_t, ldc, int64_t, batchCount, cublasComputeType_t, computeType, cublasGemmAlgo_t, algo);
DEF_FN(cublasStatus_t, cublasGemmStridedBatchedEx_64, cublasHandle_t, handle, cublasOperation_t, transa, cublasOperation_t, transb, int64_t, m, int64_t, n, int64_t, k, const void*, alpha, const void*, A, cudaDataType, Atype, int64_t, lda, long long int, strideA, const void*, B, cudaDataType, Btype, int64_t, ldb, long long int, strideB, const void*, beta, void*, C, cudaDataType, Ctype, int64_t, ldc, long long int, strideC, int64_t, batchCount, cublasComputeType_t, computeType, cublasGemmAlgo_t, algo);
DEF_FN(cublasStatus_t, cublasSgeam, cublasHandle_t, handle, cublasOperation_t, transa, cublasOperation_t, transb, int, m, int, n, const float*, alpha, const float*, A, int, lda, const float*, beta, const float*, B, int, ldb, float*, C, int, ldc);
DEF_FN(cublasStatus_t, cublasSgeam_64, cublasHandle_t, handle, cublasOperation_t, transa, cublasOperation_t, transb, int64_t, m, int64_t, n, const float*, alpha, const float*, A, int64_t, lda, const float*, beta, const float*, B, int64_t, ldb, float*, C, int64_t, ldc);
DEF_FN(cublasStatus_t, cublasDgeam, cublasHandle_t, handle, cublasOperation_t, transa, cublasOperation_t, transb, int, m, int, n, const double*, alpha, const double*, A, int, lda, const double*, beta, const double*, B, int, ldb, double*, C, int, ldc);
DEF_FN(cublasStatus_t, cublasDgeam_64, cublasHandle_t, handle, cublasOperation_t, transa, cublasOperation_t, transb, int64_t, m, int64_t, n, const double*, alpha, const double*, A, int64_t, lda, const double*, beta, const double*, B, int64_t, ldb, double*, C, int64_t, ldc);
DEF_FN(cublasStatus_t, cublasCgeam, cublasHandle_t, handle, cublasOperation_t, transa, cublasOperation_t, transb, int, m, int, n, const cuComplex*, alpha, const cuComplex*, A, int, lda, const cuComplex*, beta, const cuComplex*, B, int, ldb, cuComplex*, C, int, ldc);
DEF_FN(cublasStatus_t, cublasCgeam_64, cublasHandle_t, handle, cublasOperation_t, transa, cublasOperation_t, transb, int64_t, m, int64_t, n, const cuComplex*, alpha, const cuComplex*, A, int64_t, lda, const cuComplex*, beta, const cuComplex*, B, int64_t, ldb, cuComplex*, C, int64_t, ldc);
DEF_FN(cublasStatus_t, cublasZgeam, cublasHandle_t, handle, cublasOperation_t, transa, cublasOperation_t, transb, int, m, int, n, const cuDoubleComplex*, alpha, const cuDoubleComplex*, A, int, lda, const cuDoubleComplex*, beta, const cuDoubleComplex*, B, int, ldb, cuDoubleComplex*, C, int, ldc);
DEF_FN(cublasStatus_t, cublasZgeam_64, cublasHandle_t, handle, cublasOperation_t, transa, cublasOperation_t, transb, int64_t, m, int64_t, n, const cuDoubleComplex*, alpha, const cuDoubleComplex*, A, int64_t, lda, const cuDoubleComplex*, beta, const cuDoubleComplex*, B, int64_t, ldb, cuDoubleComplex*, C, int64_t, ldc);
DEF_FN(cublasStatus_t, cublasStrsmBatched, cublasHandle_t, handle, cublasSideMode_t, side, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int, m, int, n, const float*, alpha, const float* const*,  A, int, lda, float* const*,  B, int, ldb, int, batchCount);
DEF_FN(cublasStatus_t, cublasStrsmBatched_64, cublasHandle_t, handle, cublasSideMode_t, side, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int64_t, m, int64_t, n, const float*, alpha, const float* const*,  A, int64_t, lda, float* const*,  B, int64_t, ldb, int64_t, batchCount);
DEF_FN(cublasStatus_t, cublasDtrsmBatched, cublasHandle_t, handle, cublasSideMode_t, side, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int, m, int, n, const double*, alpha, const double* const*,  A, int, lda, double* const*,  B, int, ldb, int, batchCount);
DEF_FN(cublasStatus_t, cublasDtrsmBatched_64, cublasHandle_t, handle, cublasSideMode_t, side, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int64_t, m, int64_t, n, const double*, alpha, const double* const*,  A, int64_t, lda, double* const*,  B, int64_t, ldb, int64_t, batchCount);
DEF_FN(cublasStatus_t, cublasCtrsmBatched, cublasHandle_t, handle, cublasSideMode_t, side, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int, m, int, n, const cuComplex*, alpha, const cuComplex* const*,  A, int, lda, cuComplex* const*,  B, int, ldb, int, batchCount);
DEF_FN(cublasStatus_t, cublasCtrsmBatched_64, cublasHandle_t, handle, cublasSideMode_t, side, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int64_t, m, int64_t, n, const cuComplex*, alpha, const cuComplex* const*,  A, int64_t, lda, cuComplex* const*,  B, int64_t, ldb, int64_t, batchCount);
DEF_FN(cublasStatus_t, cublasZtrsmBatched, cublasHandle_t, handle, cublasSideMode_t, side, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int, m, int, n, const cuDoubleComplex*, alpha, const cuDoubleComplex* const*,  A, int, lda, cuDoubleComplex* const*,  B, int, ldb, int, batchCount);
DEF_FN(cublasStatus_t, cublasZtrsmBatched_64, cublasHandle_t, handle, cublasSideMode_t, side, cublasFillMode_t, uplo, cublasOperation_t, trans, cublasDiagType_t, diag, int64_t, m, int64_t, n, const cuDoubleComplex*, alpha, const cuDoubleComplex* const*,  A, int64_t, lda, cuDoubleComplex* const*,  B, int64_t, ldb, int64_t, batchCount);
DEF_FN(cublasStatus_t, cublasSdgmm, cublasHandle_t, handle, cublasSideMode_t, mode, int, m, int, n, const float*, A, int, lda, const float*, x, int, incx, float*, C, int, ldc);
DEF_FN(cublasStatus_t, cublasSdgmm_64, cublasHandle_t, handle, cublasSideMode_t, mode, int64_t, m, int64_t, n, const float*, A, int64_t, lda, const float*, x, int64_t, incx, float*, C, int64_t, ldc);
DEF_FN(cublasStatus_t, cublasDdgmm, cublasHandle_t, handle, cublasSideMode_t, mode, int, m, int, n, const double*, A, int, lda, const double*, x, int, incx, double*, C, int, ldc);
DEF_FN(cublasStatus_t, cublasDdgmm_64, cublasHandle_t, handle, cublasSideMode_t, mode, int64_t, m, int64_t, n, const double*, A, int64_t, lda, const double*, x, int64_t, incx, double*, C, int64_t, ldc);
DEF_FN(cublasStatus_t, cublasCdgmm, cublasHandle_t, handle, cublasSideMode_t, mode, int, m, int, n, const cuComplex*, A, int, lda, const cuComplex*, x, int, incx, cuComplex*, C, int, ldc);
DEF_FN(cublasStatus_t, cublasCdgmm_64, cublasHandle_t, handle, cublasSideMode_t, mode, int64_t, m, int64_t, n, const cuComplex*, A, int64_t, lda, const cuComplex*, x, int64_t, incx, cuComplex*, C, int64_t, ldc);
DEF_FN(cublasStatus_t, cublasZdgmm, cublasHandle_t, handle, cublasSideMode_t, mode, int, m, int, n, const cuDoubleComplex*, A, int, lda, const cuDoubleComplex*, x, int, incx, cuDoubleComplex*, C, int, ldc);
DEF_FN(cublasStatus_t, cublasZdgmm_64, cublasHandle_t, handle, cublasSideMode_t, mode, int64_t, m, int64_t, n, const cuDoubleComplex*, A, int64_t, lda, const cuDoubleComplex*, x, int64_t, incx, cuDoubleComplex*, C, int64_t, ldc);
DEF_FN(cublasStatus_t, cublasSmatinvBatched, cublasHandle_t, handle, int, n, const float* const*,  A, int, lda, float* const*,  Ainv, int, lda_inv, int*, info, int, batchSize);
DEF_FN(cublasStatus_t, cublasDmatinvBatched, cublasHandle_t, handle, int, n, const double* const*,  A, int, lda, double* const*,  Ainv, int, lda_inv, int*, info, int, batchSize);
DEF_FN(cublasStatus_t, cublasCmatinvBatched, cublasHandle_t, handle, int, n, const cuComplex* const*,  A, int, lda, cuComplex* const*,  Ainv, int, lda_inv, int*, info, int, batchSize);
DEF_FN(cublasStatus_t, cublasZmatinvBatched, cublasHandle_t, handle, int, n, const cuDoubleComplex* const*,  A, int, lda, cuDoubleComplex* const*,  Ainv, int, lda_inv, int*, info, int, batchSize);
DEF_FN(cublasStatus_t, cublasSgeqrfBatched, cublasHandle_t, handle, int, m, int, n, float* const*,  Aarray, int, lda, float* const*,  TauArray, int*, info, int, batchSize);
DEF_FN(cublasStatus_t, cublasDgeqrfBatched, cublasHandle_t, handle, int, m, int, n, double* const*,  Aarray, int, lda, double* const*,  TauArray, int*, info, int, batchSize);
DEF_FN(cublasStatus_t, cublasCgeqrfBatched, cublasHandle_t, handle, int, m, int, n, cuComplex* const*,  Aarray, int, lda, cuComplex* const*,  TauArray, int*, info, int, batchSize);
DEF_FN(cublasStatus_t, cublasZgeqrfBatched, cublasHandle_t, handle, int, m, int, n, cuDoubleComplex* const*,  Aarray, int, lda, cuDoubleComplex* const*,  TauArray, int*, info, int, batchSize);
DEF_FN(cublasStatus_t, cublasSgelsBatched, cublasHandle_t, handle, cublasOperation_t, trans, int, m, int, n, int, nrhs, float* const*,  Aarray, int, lda, float* const*,  Carray, int, ldc, int*, info, int*, devInfoArray, int, batchSize);
DEF_FN(cublasStatus_t, cublasDgelsBatched, cublasHandle_t, handle, cublasOperation_t, trans, int, m, int, n, int, nrhs, double* const*,  Aarray, int, lda, double* const*,  Carray, int, ldc, int*, info, int*, devInfoArray, int, batchSize);
DEF_FN(cublasStatus_t, cublasCgelsBatched, cublasHandle_t, handle, cublasOperation_t, trans, int, m, int, n, int, nrhs, cuComplex* const*,  Aarray, int, lda, cuComplex* const*,  Carray, int, ldc, int*, info, int*, devInfoArray, int, batchSize);
DEF_FN(cublasStatus_t, cublasZgelsBatched, cublasHandle_t, handle, cublasOperation_t, trans, int, m, int, n, int, nrhs, cuDoubleComplex* const*,  Aarray, int, lda, cuDoubleComplex* const*,  Carray, int, ldc, int*, info, int*, devInfoArray, int, batchSize);
DEF_FN(cublasStatus_t, cublasStpttr, cublasHandle_t, handle, cublasFillMode_t, uplo, int, n, const float*, AP, float*, A, int, lda);
DEF_FN(cublasStatus_t, cublasDtpttr, cublasHandle_t, handle, cublasFillMode_t, uplo, int, n, const double*, AP, double*, A, int, lda);
DEF_FN(cublasStatus_t, cublasCtpttr, cublasHandle_t, handle, cublasFillMode_t, uplo, int, n, const cuComplex*, AP, cuComplex*, A, int, lda);
DEF_FN(cublasStatus_t, cublasZtpttr, , cublasHandle_t, handle, cublasFillMode_t, uplo, int, n, const cuDoubleComplex*, AP, cuDoubleComplex*, A, int, lda);
DEF_FN(cublasStatus_t, cublasStrttp, cublasHandle_t, handle, cublasFillMode_t, uplo, int, n, const float*, A, int, lda, float*, AP);
DEF_FN(cublasStatus_t, cublasDtrttp, cublasHandle_t, handle, cublasFillMode_t, uplo, int, n, const double*, A, int, lda, double*, AP);
DEF_FN(cublasStatus_t, cublasCtrttp, cublasHandle_t, handle, cublasFillMode_t, uplo, int, n, const cuComplex*, A, int, lda, cuComplex*, AP);
DEF_FN(cublasStatus_t, cublasZtrttp, , cublasHandle_t, handle, cublasFillMode_t, uplo, int, n, const cuDoubleComplex*, A, int, lda, cuDoubleComplex*, AP);
DEF_FN(cublasStatus_t, cublasSgetrfBatched, cublasHandle_t, handle, int, n, float* const*,  A, int, lda, int*, P, int*, info, int, batchSize);
DEF_FN(cublasStatus_t, cublasDgetrfBatched, cublasHandle_t, handle, int, n, double* const*,  A, int, lda, int*, P, int*, info, int, batchSize);
DEF_FN(cublasStatus_t, cublasCgetrfBatched, cublasHandle_t, handle, int, n, cuComplex* const*,  A, int, lda, int*, P, int*, info, int, batchSize);
DEF_FN(cublasStatus_t, cublasZgetrfBatched, , cublasHandle_t, handle, int, n, cuDoubleComplex* const*,  A, int, lda, int*, P, int*, info, int, batchSize);
DEF_FN(cublasStatus_t, cublasSgetriBatched, cublasHandle_t, handle, int, n, const float* const*,  A, int, lda, const int*, P, float* const*,  C, int, ldc, int*, info, int, batchSize);
DEF_FN(cublasStatus_t, cublasDgetriBatched, cublasHandle_t, handle, int, n, const double* const*,  A, int, lda, const int*, P, double* const*,  C, int, ldc, int*, info, int, batchSize);
DEF_FN(cublasStatus_t, cublasCgetriBatched, cublasHandle_t, handle, int, n, const cuComplex* const*,  A, int, lda, const int*, P, cuComplex* const*,  C, int, ldc, int*, info, int, batchSize);
DEF_FN(cublasStatus_t, cublasZgetriBatched, cublasHandle_t, handle, int, n, const cuDoubleComplex* const*,  A, int, lda, const int*, P, cuDoubleComplex* const*,  C, int, ldc, int*, info, int, batchSize);
DEF_FN(cublasStatus_t, cublasSgetrsBatched, cublasHandle_t, handle, cublasOperation_t, trans, int, n, int, nrhs, const float* const*,  Aarray, int, lda, const int*, devIpiv, float* const*,  Barray, int, ldb, int*, info, int, batchSize);
DEF_FN(cublasStatus_t, cublasDgetrsBatched, cublasHandle_t, handle, cublasOperation_t, trans, int, n, int, nrhs, const double* const*,  Aarray, int, lda, const int*, devIpiv, double* const*,  Barray, int, ldb, int*, info, int, batchSize);
DEF_FN(cublasStatus_t, cublasCgetrsBatched, cublasHandle_t, handle, cublasOperation_t, trans, int, n, int, nrhs, const cuComplex* const*,  Aarray, int, lda, const int*, devIpiv, cuComplex* const*,  Barray, int, ldb, int*, info, int, batchSize);
DEF_FN(cublasStatus_t, cublasZgetrsBatched, cublasHandle_t, handle, cublasOperation_t, trans, int, n, int, nrhs, const cuDoubleComplex* const*,  Aarray, int, lda, const int*, devIpiv, cuDoubleComplex* const*,  Barray, int, ldb, int*, info, int, batchSize);

cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t handle,
                            cublasOperation_t transa,
                            cublasOperation_t transb,
                            int m,
                            int n,
                            int k,
                            const void    *alpha,
                            const void     *A,
                            cudaDataType_t Atype,
                            int lda,
                            long long int strideA,
                            const void     *B,
                            cudaDataType_t Btype,
                            int ldb,
                            long long int strideB,
                            const void    *beta,
                            void           *C,
                            cudaDataType_t Ctype,
                            int ldc,
                            long long int strideC,
                            int batchCount,
                            cublasComputeType_t computeType,
                            cublasGemmAlgo_t algo)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval_1;
    retval_1 = rpc_cublasgemmstridedbatchedex_1(
        (ptr)handle,
        (int)transa,
        (int)transb,
        m, n, k,
        *((float*)alpha),
        (ptr)A,
        (int)Atype,
        lda,
        strideA,
        (ptr)B,
        (int)Btype,
        ldb,
        strideB,
        *((float*)beta),
        (ptr)C,
        (int)Ctype,
        ldc,
        strideC,
        batchCount,
        (int)computeType,
        (int)algo,
        &result, clnt
    );
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    return result;
}


cublasStatus_t cublasGemmEx(cublasHandle_t handle,
                           cublasOperation_t transa,
                           cublasOperation_t transb,
                           int m,
                           int n,
                           int k,
                           const void    *alpha,
                           const void     *A,
                           cudaDataType_t Atype,
                           int lda,
                           const void     *B,
                           cudaDataType_t Btype,
                           int ldb,
                           const void    *beta,
                           void           *C,
                           cudaDataType_t Ctype,
                           int ldc,
                           cublasComputeType_t computeType,
                           cublasGemmAlgo_t algo)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval_1;
    retval_1 = rpc_cublasgemmex_1(
        (ptr)handle,
        (int)transa,
        (int)transb,
        m, n, k,
        *((float*)alpha),
        (ptr)A, (int)Atype, lda,
        (ptr)B, (int)Btype, ldb,
        *((float*)beta),
        (ptr)C, (int)Ctype, ldc,
        computeType, algo,
         &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    return result;
}


cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t handle,
                                  cublasOperation_t transa,
                                  cublasOperation_t transb,
                                  int m, int n, int k,
                                  const float           *alpha,
                                  const float           *A, int lda,
                                  long long int          strideA,
                                  const float           *B, int ldb,
                                  long long int          strideB,
                                  const float           *beta,
                                  float                 *C, int ldc,
                                  long long int          strideC,
                                  int batchCount)
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT

    int result;
    enum clnt_stat retval_1;
    retval_1 = rpc_cublasgemmstridedbatched_1(
        (ptr)handle,
        (int)transa,
        (int)transb,
        m, n, k,
        *alpha,
        (ptr)A,
        lda,
        strideA,
        (ptr)B,
        ldb,
	strideB,
        *beta,
        (ptr)C,
        ldc,
        strideC,
        batchCount,
        &result, clnt
    );
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    return result;
}

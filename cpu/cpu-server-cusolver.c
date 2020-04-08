#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>

//for strerror
#include <string.h>
#include <errno.h>

#include "cpu_rpc_prot.h"
#include "cpu-common.h"
#include "cpu-utils.h"
#include "log.h"

bool_t rpc_cusolverdncreate_1_svc(ptr_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cusolverDnCreate\n");
    result->err = cusolverDnCreate((cusolverDnHandle_t*)&result->ptr_result_u.ptr);
    return 1;
}

bool_t rpc_cusolverdnsetstream_1_svc(ptr handle, ptr stream, int *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cusolverDnSetStream\n");
    *result = cusolverDnSetStream((cusolverDnHandle_t)handle, (cudaStream_t)stream);
    return 1;
}

bool_t rpc_cusolverdndgetrf_buffersize_1_svc(ptr handle, int m, int n, ptr A, int lda, int_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cusolverDnDgetrf_buffersize\n");
    result->err = cusolverDnDgetrf_bufferSize((cusolverDnHandle_t)handle, m, n, (double*)A, lda, &result->int_result_u.data);
    return 1;
}

bool_t rpc_cusolverdndgetrf_1_svc(ptr handle, int m, int n, ptr A, int lda, ptr Workspace, ptr devIpiv, ptr devInfo, int *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cusolverDnDgetrf\n");
    *result = cusolverDnDgetrf((cusolverDnHandle_t)handle, m, n, (double*)A, lda,
                               (double*)Workspace, (int*)devIpiv, (int*)devInfo);
    return 1;
}

bool_t rpc_cusolverdndgetrs_1_svc(ptr handle, int trans, int n, int nrhs, ptr A, int lda, ptr devIpiv, ptr B, int ldb, ptr devInfo, int *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cusolverDnDgetrs\n");
    *result = cusolverDnDgetrs((cusolverDnHandle_t)handle, (cublasOperation_t)trans, n, nrhs,
                               (double*)A, lda, (int*)devIpiv, (double*)B, ldb, (int*)devInfo);
    return 1;
}

bool_t rpc_cusolverdndestroy_1_svc(ptr handle, int *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cusolverDnDestroy\n");
    *result = cusolverDnDestroy((cusolverDnHandle_t)handle);
    return 1;
}

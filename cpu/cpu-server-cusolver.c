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
#include "resource-mg.h"
#define WITH_RECORDER
#include "api-recorder.h"
#include "cpu-server-cusolver.h"



int cusolver_init(int bypass, resource_mg *streams, resource_mg *memory)
{
    int ret = 0;
    ret &= resource_mg_init(&rm_cusolver, bypass);
    return ret;
}

resource_mg *cusolver_get_rm(void)
{
    return &rm_cusolver;
}

int cusolver_deinit(void)
{
    resource_mg_free(&rm_cusolver);
    return 0;

}

bool_t rpc_cusolverdncreate_1_svc(ptr_result *result, struct svc_req *rqstp)
{
    RECORD_VOID_API;
    LOGE(LOG_DEBUG, "cusolverDnCreate");

    result->err = cusolverDnCreate((cusolverDnHandle_t*)&result->ptr_result_u.ptr);
    RECORD_RESULT(ptr_result_u, *result);
    resource_mg_create(&rm_cusolver, (void*)result->ptr_result_u.ptr);
    return 1;
}

bool_t rpc_cusolverdnsetstream_1_svc(ptr handle, ptr stream, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cusolverdnsetstream_1_argument);
    RECORD_ARG(1, handle);
    RECORD_ARG(2, stream);
    LOGE(LOG_DEBUG, "cusolverDnSetStream");
    *result = cusolverDnSetStream(resource_mg_get(&rm_cusolver, (void*)handle),
                                  resource_mg_get(&rm_streams, (void*)stream));
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cusolverdndgetrf_buffersize_1_svc(ptr handle, int m, int n, ptr A, int lda, int_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cusolverDnDgetrf_buffersize");
    result->err = cusolverDnDgetrf_bufferSize(resource_mg_get(&rm_cusolver, (void*)handle),
                                              m, n,
                                              resource_mg_get(&rm_memory, (void*)A),
                                              lda, &result->int_result_u.data);
    return 1;
}

bool_t rpc_cusolverdndgetrf_1_svc(ptr handle, int m, int n, ptr A, int lda, ptr Workspace, ptr devIpiv, ptr devInfo, int *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cusolverDnDgetrf");
    *result = cusolverDnDgetrf(resource_mg_get(&rm_cusolver, (void*)handle),
                               m, n,
                               resource_mg_get(&rm_memory, (void*)A),
                               lda,
                               resource_mg_get(&rm_memory, (void*)Workspace),
                               resource_mg_get(&rm_memory, (void*)devIpiv),
                               resource_mg_get(&rm_memory, (void*)devInfo));
    return 1;
}

bool_t rpc_cusolverdndgetrs_1_svc(ptr handle, int trans, int n, int nrhs, ptr A, int lda, ptr devIpiv, ptr B, int ldb, ptr devInfo, int *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cusolverDnDgetrs");
    *result = cusolverDnDgetrs(resource_mg_get(&rm_cusolver, (void*)handle),
                               (cublasOperation_t)trans, n, nrhs,
                               resource_mg_get(&rm_memory, (void*)A),
                               lda,
                               resource_mg_get(&rm_memory, (void*)devIpiv),
                               resource_mg_get(&rm_memory, (void*)B),
                               ldb,
                               resource_mg_get(&rm_memory, (void*)devInfo));

    LOGE(LOG_DEBUG, "handle: %p, A: %p, devIpiv: %p, B: %p, devInfo: %p", handle, A, devIpiv, B, devInfo);
    LOGE(LOG_DEBUG, "trans: %d, n: %d, nrhs: %d, lda: %d, ldb: %d, result: %d", trans, n, nrhs, lda, ldb, *result);
    return 1;
}

bool_t rpc_cusolverdndestroy_1_svc(ptr handle, int *result, struct svc_req *rqstp)
{
    RECORD_API(ptr);
    RECORD_SINGLE_ARG(handle);
    LOGE(LOG_DEBUG, "cusolverDnDestroy");
    *result = cusolverDnDestroy(resource_mg_get(&rm_cusolver, (void*)handle));
    RECORD_RESULT(integer, *result);
    return 1;
}

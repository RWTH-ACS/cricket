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


static resource_mg rm_cublas;
static resource_mg *rm_memory;

int cublas_init(int bypass, resource_mg *memory)
{
    int ret = 0;
    ret &= resource_mg_init(&rm_cublas, bypass);
    rm_memory = memory;
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

    result->err = cublasCreate_v2((cublasHandle_t*)&result->ptr_result_u.ptr);
    RECORD_RESULT(ptr_result_u, *result);
    resource_mg_create(&rm_cublas, (void*)result->ptr_result_u.ptr);
    return 1;
}

bool_t rpc_cublasdgemm_1_svc(ptr handle, int transa, int transb, int m, int n, int k, double alpha,
            ptr A, int lda,
            ptr B, int ldb, double beta,
            ptr C, int ldc,
            int *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cublasDgemm");
    *result = cublasDgemm(resource_mg_get(&rm_cublas, (void*)handle),
                    (cublasOperation_t) transa,
                    (cublasOperation_t) transb,
                    m, n, k, &alpha,
                    resource_mg_get(&rm_cublas, (void*)A), lda,
                    resource_mg_get(&rm_cublas, (void*)B), ldb, &beta,
                    resource_mg_get(&rm_cublas, (void*)C), ldc
    );
    return 1;
}

bool_t rpc_cublasdestroy_1_svc(ptr handle, int *result, struct svc_req *rqstp)
{
    RECORD_API(ptr);
    RECORD_SINGLE_ARG(handle);
    LOGE(LOG_DEBUG, "cublasDestroy_v2");
    *result = cublasDestroy_v2(resource_mg_get(&rm_cublas, (void*)handle));
    RECORD_RESULT(integer, *result);
    return 1;
}

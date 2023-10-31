#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublasLt.h>

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


int cublaslt_init(int bypass, resource_mg *memory)
{
    int ret = 0;
    ret &= resource_mg_init(&rm_cublaslt, bypass);
    return ret;
}

int cublaslt_deinit(void)
{
    resource_mg_free(&rm_cublaslt);
    return 0;
}

bool_t rpc_cublasltcreate_1_svc(ptr_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cublasLtCreate");

    GSCHED_RETAIN;
    result->err = cublasLtCreate((cublasLtHandle_t*)&result->ptr_result_u.ptr);
    resource_mg_create(&rm_cublaslt, (void*)result->ptr_result_u.ptr);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cublasltmatrixlayoutcreate_1_svc(int type, uint64_t row, uint64_t cols, int64_t ld, ptr_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cublasLtMatmulLayoutCreate");

    GSCHED_RETAIN;
    result-> err = cublasLtMatrixLayoutCreate(
        (cublasLtMatrixLayout_t*)&result->ptr_result_u.ptr,
        (cudaDataType)type,
        row,
        cols,
        ld
    );
    if (resource_mg_create(&rm_cublaslt, (void*)result->ptr_result_u.ptr) != 0) {
        LOGE(LOG_ERROR, "error in resource manager");
    }
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cublasltmatmulpreferencecreate_1_svc(ptr_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cublasLtMatmulPreferenceCreate");

    GSCHED_RETAIN;
    result-> err = cublasLtMatmulPreferenceCreate(
        (cublasLtMatmulPreference_t*)&result->ptr_result_u.ptr
    );
    if (resource_mg_create(&rm_cublaslt, (void*)result->ptr_result_u.ptr) != 0) {
        LOGE(LOG_ERROR, "error in resource manager");
    }
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cublasltmatmuldesccreate_1_svc(int computeType, int scaleType, ptr_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cublasLtMatmulDescCreate");

    GSCHED_RETAIN;
    result->err = cublasLtMatmulDescCreate((cublasLtMatmulDesc_t*)&result->ptr_result_u.ptr,
     (cublasComputeType_t)computeType,
     (cudaDataType_t)scaleType);
    if (resource_mg_create(&rm_cublaslt, (void*)result->ptr_result_u.ptr) != 0) {
        LOGE(LOG_ERROR, "error in resource manager");
    }
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cublasltmatmuldescdestroy_1_svc(ptr matmulDesc, int *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cublasLtMatmulDescDestroy");

    GSCHED_RETAIN;
    *result = cublasLtMatmulDescDestroy(resource_mg_get(&rm_cublaslt, (void*)matmulDesc));
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cublasltmatmulpreferencedestroy_1_svc(ptr pref, int *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cublasLtMatmulPreferenceDestroy");

    GSCHED_RETAIN;
    *result = cublasLtMatmulPreferenceDestroy(resource_mg_get(&rm_cublaslt, (void*)pref));
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cublasltmatrixlayoutdestroy_1_svc(ptr matLayout, int *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cublasLtMatrixLayoutDestroy");

    GSCHED_RETAIN;
    *result = cublasLtMatrixLayoutDestroy(resource_mg_get(&rm_cublaslt, (void*)matLayout));
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cublasltmatmulalgogetheuristic_1_svc(ptr lightHandle, ptr operationDesc, ptr Adesc, ptr Bdesc, ptr Cdesc, ptr Ddesc, ptr preference, int requestedAlgoCount, matmul_hr_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cublasLtMatmulAlgoGetHeuristic");
    GSCHED_RETAIN;

    if (sizeof(result->matmul_hr_result_u.data.p) != sizeof(cublasLtMatmulHeuristicResult_t)) {
	LOGE(LOG_ERROR, "cublasLtMatmulHeuristicResult_t size mismatch");
        return 0;
     }

    result->err = cublasLtMatmulAlgoGetHeuristic(
        (cublasLtHandle_t)resource_mg_get(&rm_cublaslt, (void*)lightHandle),
        (cublasLtMatmulDesc_t)resource_mg_get(&rm_cublaslt, (void*)operationDesc),
        (cublasLtMatrixLayout_t)resource_mg_get(&rm_cublaslt, (void*)Adesc),
        (cublasLtMatrixLayout_t)resource_mg_get(&rm_cublaslt, (void*)Bdesc),
        (cublasLtMatrixLayout_t)resource_mg_get(&rm_cublaslt, (void*)Cdesc),
        (cublasLtMatrixLayout_t)resource_mg_get(&rm_cublaslt, (void*)Ddesc),
        (cublasLtMatmulPreference_t)resource_mg_get(&rm_cublaslt, (void*)preference),
        requestedAlgoCount,
        (void*)&result->matmul_hr_result_u.data.p,
        &result->matmul_hr_result_u.data.s);

    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cublasltmatmuldescsetattribute_1_svc(ptr matmulDesc, int attr, mem_data data, int *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cublasLtMatmulDescSetAttribute");
    GSCHED_RETAIN;

    *result = cublasLtMatmulDescSetAttribute(
        (cublasLtMatmulDesc_t)resource_mg_get(&rm_cublaslt, (void*)matmulDesc),
        (cublasLtMatmulDescAttributes_t)attr,
        data.mem_data_val,
        data.mem_data_len
    );

    GSCHED_RELEASE;
    return 1;
}


bool_t rpc_cublasltmatmul_1_svc(ptr lightHandle,
    ptr computeDesc,
    float alpha,
    ptr A,
    ptr Adesc,
    ptr B,
    ptr Bdesc,
    float beta,
    ptr C,
    ptr Cdesc,
    ptr D,
    ptr Ddesc,
    ptr algo,
    ptr workspace,
    size_t workspaceSizeInBytes,
    ptr stream,
    int *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cublasLtMatmul");
    GSCHED_RETAIN;

    *result = cublasLtMatmul(
        (cublasLtHandle_t)resource_mg_get(&rm_cublaslt, (void*)lightHandle),
        (cublasLtMatmulDesc_t)resource_mg_get(&rm_cublaslt, (void*)computeDesc),
        &alpha,
        resource_mg_get(&rm_memory, (void*)A),
        (cublasLtMatrixLayout_t)resource_mg_get(&rm_cublaslt, (void*)Adesc),
        resource_mg_get(&rm_memory, (void*)B),
        (cublasLtMatrixLayout_t)resource_mg_get(&rm_cublaslt, (void*)Bdesc),
        &beta,
        resource_mg_get(&rm_memory, (void*)C),
        (cublasLtMatrixLayout_t)resource_mg_get(&rm_cublaslt, (void*)Cdesc),
        resource_mg_get(&rm_memory, (void*)D),
        (cublasLtMatrixLayout_t)resource_mg_get(&rm_cublaslt, (void*)Ddesc),
        // (const cublasLtMatmulAlgo_t *)algo,
	NULL,
        resource_mg_get(&rm_memory, (void*)workspace),
        workspaceSizeInBytes,
       (cudaStream_t)resource_mg_get(&rm_streams, (void*)stream)
    );

    GSCHED_RELEASE;
    return 1;
}

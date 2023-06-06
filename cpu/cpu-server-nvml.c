#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>

#include <nvml.h>
#include <cuda_runtime_api.h>

#include "cpu_rpc_prot.h"
#include "cpu-common.h"
#include "cpu-utils.h"
#include "log.h"
#include "resource-mg.h"
#define WITH_RECORDER
#include "api-recorder.h"
#include "gsched.h"

int server_nvml_init(int restore)
{
    int ret = 0;
    if (!restore) {
        //ret &= resource_mg_init(&rm_modules, 1);
    } else {
        //ret &= resource_mg_init(&rm_modules, 0);
        //ret &= server_driver_restore("ckp");
    }
    return ret;
}

int server_nvml_deinit(void)
{
    //resource_mg_free(&rm_modules);
    return 0;
}

bool_t rpc_nvmldevicegetcount_v2_1_svc(int_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    // Workaround for pytorch expecting nvmlDeviceGetCount and cudaGetDeviceCount to be the same
    //result->err = nvmlDeviceGetCount_v2(&result->int_result_u.data);
    result->err = cudaGetDeviceCount(&result->int_result_u.data);
    LOGE(LOG_DEBUG, "%s: %d", __FUNCTION__, result->int_result_u.data);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_nvmlinitwithflags_1_svc(int flags, int *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    *result = nvmlInitWithFlags(flags);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_nvmlinit_v2_1_svc(int *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    *result = nvmlInit_v2();
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_nvmlshutdown_1_svc(int *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    *result = nvmlShutdown();
    GSCHED_RELEASE;
    return 1;
}
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <nvml.h>

#include "cpu-libwrap.h"
#include "cpu_rpc_prot.h"
#include "cpu-common.h"
#include "cpu-utils.h"
#include "log.h"

#ifdef WITH_API_CNT
static int api_call_cnt = 0;
void cpu_nvml_print_api_call_cnt(void)
{
    LOG(LOG_INFO, "nvml api-call-cnt: %d", api_call_cnt);
}
#endif //WITH_API_CNT

nvmlReturn_t nvmlInitWithFlags ( unsigned int  flags )
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval_1;
    retval_1 = rpc_nvmlinitwithflags_1(flags, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        LOGE(LOG_ERROR, "call failed: %s", __FUNCTION__);
        return result;
    }
    return result;
}

#undef nvmlInit
nvmlReturn_t nvmlInit(void)
{
    return nvmlInitWithFlags(0);
}

nvmlReturn_t nvmlInit_v2 ( void )
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval_1;
    retval_1 = rpc_nvmlinit_v2_1(&result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        LOGE(LOG_ERROR, "call failed: %s", __FUNCTION__);
        return result;
    }
    return result;
}
nvmlReturn_t nvmlShutdown ( void )
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int result;
    enum clnt_stat retval_1;
    retval_1 = rpc_nvmlshutdown_1(&result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        LOGE(LOG_ERROR, "call failed: %s", __FUNCTION__);
        return result;
    }
    return result;
}


DEF_FN(nvmlReturn_t, nvmlDeviceGetAPIRestriction, nvmlDevice_t, device, nvmlRestrictedAPI_t, apiType, nvmlEnableState_t*, isRestricted )
DEF_FN(nvmlReturn_t, nvmlDeviceGetAdaptiveClockInfoStatus, nvmlDevice_t, device, unsigned int*, adaptiveClockStatus )
DEF_FN(nvmlReturn_t, nvmlDeviceGetApplicationsClock, nvmlDevice_t, device, nvmlClockType_t, clockType, unsigned int*, clockMHz )
#if NVML_API_VERSION >= 12
DEF_FN(nvmlReturn_t, nvmlDeviceGetArchitecture, nvmlDevice_t, device, nvmlDeviceArchitecture_t*, arch )
DEF_FN(nvmlReturn_t, nvmlDeviceGetAttributes_v2, nvmlDevice_t, device, nvmlDeviceAttributes_t*, attributes )
#endif
DEF_FN(nvmlReturn_t, nvmlDeviceGetAutoBoostedClocksEnabled, nvmlDevice_t, device, nvmlEnableState_t*, isEnabled, nvmlEnableState_t*, defaultIsEnabled )
DEF_FN(nvmlReturn_t, nvmlDeviceGetBAR1MemoryInfo, nvmlDevice_t, device, nvmlBAR1Memory_t*, bar1Memory )
DEF_FN(nvmlReturn_t, nvmlDeviceGetBoardId, nvmlDevice_t, device, unsigned int*, boardId )
DEF_FN(nvmlReturn_t, nvmlDeviceGetBoardPartNumber, nvmlDevice_t, device, char*, partNumber, unsigned int,  length )
DEF_FN(nvmlReturn_t, nvmlDeviceGetBrand, nvmlDevice_t, device, nvmlBrandType_t*, type )
DEF_FN(nvmlReturn_t, nvmlDeviceGetBridgeChipInfo, nvmlDevice_t, device, nvmlBridgeChipHierarchy_t*, bridgeHierarchy )
DEF_FN(nvmlReturn_t, nvmlDeviceGetClock, nvmlDevice_t, device, nvmlClockType_t, clockType, nvmlClockId_t, clockId, unsigned int*, clockMHz )
DEF_FN(nvmlReturn_t, nvmlDeviceGetClockInfo, nvmlDevice_t, device, nvmlClockType_t, type, unsigned int*, clock )
DEF_FN(nvmlReturn_t, nvmlDeviceGetComputeMode, nvmlDevice_t, device, nvmlComputeMode_t*, mode )
DEF_FN(nvmlReturn_t, nvmlDeviceGetComputeRunningProcesses_v3, nvmlDevice_t, device, unsigned int*, infoCount, nvmlProcessInfo_t*, infos )
nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int* deviceCount )
{
#ifdef WITH_API_CNT
    api_call_cnt++;
#endif //WITH_API_CNT
    int_result result;
    enum clnt_stat retval_1;
    if (deviceCount == NULL) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    retval_1 = rpc_nvmldevicegetcount_v2_1(&result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        LOGE(LOG_ERROR, "call failed: %s", __FUNCTION__);
    }
    if (result.err == 0) {
        *deviceCount = result.int_result_u.data;
    }
    return result.err;
}
DEF_FN(nvmlReturn_t, nvmlDeviceGetCudaComputeCapability, nvmlDevice_t, device, int*, major, int*, minor )
DEF_FN(nvmlReturn_t, nvmlDeviceGetCurrPcieLinkGeneration, nvmlDevice_t, device, unsigned int*, currLinkGen )
DEF_FN(nvmlReturn_t, nvmlDeviceGetCurrPcieLinkWidth, nvmlDevice_t, device, unsigned int*, currLinkWidth )
DEF_FN(nvmlReturn_t, nvmlDeviceGetCurrentClocksThrottleReasons, nvmlDevice_t, device, unsigned long long*, clocksThrottleReasons )
DEF_FN(nvmlReturn_t, nvmlDeviceGetDecoderUtilization, nvmlDevice_t, device, unsigned int*, utilization, unsigned int*, samplingPeriodUs )
DEF_FN(nvmlReturn_t, nvmlDeviceGetDefaultApplicationsClock, nvmlDevice_t, device, nvmlClockType_t, clockType, unsigned int*, clockMHz )
DEF_FN(nvmlReturn_t, nvmlDeviceGetDefaultEccMode, nvmlDevice_t, device, nvmlEnableState_t*, defaultMode )
DEF_FN(nvmlReturn_t, nvmlDeviceGetDetailedEccErrors, nvmlDevice_t, device, nvmlMemoryErrorType_t, errorType, nvmlEccCounterType_t, counterType, nvmlEccErrorCounts_t*, eccCounts )
DEF_FN(nvmlReturn_t, nvmlDeviceGetDisplayActive, nvmlDevice_t, device, nvmlEnableState_t*, isActive )
DEF_FN(nvmlReturn_t, nvmlDeviceGetDisplayMode, nvmlDevice_t, device, nvmlEnableState_t*, display )
DEF_FN(nvmlReturn_t, nvmlDeviceGetDriverModel, nvmlDevice_t, device, nvmlDriverModel_t*, current, nvmlDriverModel_t*, pending )
DEF_FN(nvmlReturn_t, nvmlDeviceGetEccMode, nvmlDevice_t, device, nvmlEnableState_t*, current, nvmlEnableState_t*, pending )
DEF_FN(nvmlReturn_t, nvmlDeviceGetEncoderCapacity, nvmlDevice_t, device, nvmlEncoderType_t, encoderQueryType, unsigned int*, encoderCapacity )
DEF_FN(nvmlReturn_t, nvmlDeviceGetEncoderSessions, nvmlDevice_t, device, unsigned int*, sessionCount, nvmlEncoderSessionInfo_t*, sessionInfos )
DEF_FN(nvmlReturn_t, nvmlDeviceGetEncoderStats, nvmlDevice_t, device, unsigned int*, sessionCount, unsigned int*, averageFps, unsigned int*, averageLatency )
DEF_FN(nvmlReturn_t, nvmlDeviceGetEncoderUtilization, nvmlDevice_t, device, unsigned int*, utilization, unsigned int*, samplingPeriodUs )
DEF_FN(nvmlReturn_t, nvmlDeviceGetEnforcedPowerLimit, nvmlDevice_t, device, unsigned int*, limit )
DEF_FN(nvmlReturn_t, nvmlDeviceGetFBCSessions, nvmlDevice_t, device, unsigned int*, sessionCount, nvmlFBCSessionInfo_t*, sessionInfo )
DEF_FN(nvmlReturn_t, nvmlDeviceGetFBCStats, nvmlDevice_t, device, nvmlFBCStats_t*, fbcStats )
#if NVML_API_VERSION >= 12
DEF_FN(nvmlReturn_t, nvmlDeviceGetFanControlPolicy_v2, nvmlDevice_t, device, unsigned int,  fan, nvmlFanControlPolicy_t*, policy )
#endif
DEF_FN(nvmlReturn_t, nvmlDeviceGetFanSpeed, nvmlDevice_t, device, unsigned int*, speed )
DEF_FN(nvmlReturn_t, nvmlDeviceGetFanSpeed_v2, nvmlDevice_t, device, unsigned int,  fan, unsigned int*, speed )
DEF_FN(nvmlReturn_t, nvmlDeviceGetGpuMaxPcieLinkGeneration, nvmlDevice_t, device, unsigned int*, maxLinkGenDevice )
DEF_FN(nvmlReturn_t, nvmlDeviceGetGpuOperationMode, nvmlDevice_t, device, nvmlGpuOperationMode_t*, current, nvmlGpuOperationMode_t*, pending )
DEF_FN(nvmlReturn_t, nvmlDeviceGetGraphicsRunningProcesses_v3, nvmlDevice_t, device, unsigned int*, infoCount, nvmlProcessInfo_t*, infos )
DEF_FN(nvmlReturn_t, nvmlDeviceGetHandleByIndex_v2, unsigned int,  index, nvmlDevice_t*, device )
DEF_FN(nvmlReturn_t, nvmlDeviceGetHandleByPciBusId_v2, const char*, pciBusId, nvmlDevice_t*, device )
DEF_FN(nvmlReturn_t, nvmlDeviceGetHandleBySerial, const char*, serial, nvmlDevice_t*, device )
DEF_FN(nvmlReturn_t, nvmlDeviceGetHandleByUUID, const char*, uuid, nvmlDevice_t*, device )
DEF_FN(nvmlReturn_t, nvmlDeviceGetIndex, nvmlDevice_t, device, unsigned int*, index )
DEF_FN(nvmlReturn_t, nvmlDeviceGetInforomConfigurationChecksum, nvmlDevice_t, device, unsigned int*, checksum )
DEF_FN(nvmlReturn_t, nvmlDeviceGetInforomImageVersion, nvmlDevice_t, device, char*, version, unsigned int,  length )
DEF_FN(nvmlReturn_t, nvmlDeviceGetInforomVersion, nvmlDevice_t, device, nvmlInforomObject_t, object, char*, version, unsigned int,  length )
DEF_FN(nvmlReturn_t, nvmlDeviceGetIrqNum, nvmlDevice_t, device, unsigned int*, irqNum )
DEF_FN(nvmlReturn_t, nvmlDeviceGetMPSComputeRunningProcesses_v3, nvmlDevice_t, device, unsigned int*, infoCount, nvmlProcessInfo_t*, infos )
DEF_FN(nvmlReturn_t, nvmlDeviceGetMaxClockInfo, nvmlDevice_t, device, nvmlClockType_t, type, unsigned int*, clock )
DEF_FN(nvmlReturn_t, nvmlDeviceGetMaxCustomerBoostClock, nvmlDevice_t, device, nvmlClockType_t, clockType, unsigned int*, clockMHz )
DEF_FN(nvmlReturn_t, nvmlDeviceGetMaxPcieLinkGeneration, nvmlDevice_t, device, unsigned int*, maxLinkGen )
DEF_FN(nvmlReturn_t, nvmlDeviceGetMaxPcieLinkWidth, nvmlDevice_t, device, unsigned int*, maxLinkWidth )
DEF_FN(nvmlReturn_t, nvmlDeviceGetMemoryBusWidth, nvmlDevice_t, device, unsigned int*, busWidth )
DEF_FN(nvmlReturn_t, nvmlDeviceGetMemoryErrorCounter, nvmlDevice_t, device, nvmlMemoryErrorType_t, errorType, nvmlEccCounterType_t, counterType, nvmlMemoryLocation_t, locationType, unsigned long long*, count )
DEF_FN(nvmlReturn_t, nvmlDeviceGetMemoryInfo, nvmlDevice_t, device, nvmlMemory_t*, memory )
DEF_FN(nvmlReturn_t, nvmlDeviceGetMinMaxFanSpeed, nvmlDevice_t, device, unsigned int*, minSpeed, unsigned int*, maxSpeed )
DEF_FN(nvmlReturn_t, nvmlDeviceGetMinorNumber, nvmlDevice_t, device, unsigned int*, minorNumber )
DEF_FN(nvmlReturn_t, nvmlDeviceGetMultiGpuBoard, nvmlDevice_t, device, unsigned int*, multiGpuBool )
DEF_FN(nvmlReturn_t, nvmlDeviceGetName, nvmlDevice_t, device, char*, name, unsigned int,  length )
DEF_FN(nvmlReturn_t, nvmlDeviceGetNumFans, nvmlDevice_t, device, unsigned int*, numFans )
DEF_FN(nvmlReturn_t, nvmlDeviceGetNumGpuCores, nvmlDevice_t, device, unsigned int*, numCores )
DEF_FN(nvmlReturn_t, nvmlDeviceGetP2PStatus, nvmlDevice_t, device1, nvmlDevice_t, device2, nvmlGpuP2PCapsIndex_t, p2pIndex, nvmlGpuP2PStatus_t*, p2pStatus )
DEF_FN(nvmlReturn_t, nvmlDeviceGetPciInfo_v3, nvmlDevice_t, device, nvmlPciInfo_t*, pci )
DEF_FN(nvmlReturn_t, nvmlDeviceGetPcieLinkMaxSpeed, nvmlDevice_t, device, unsigned int*, maxSpeed )
DEF_FN(nvmlReturn_t, nvmlDeviceGetPcieReplayCounter, nvmlDevice_t, device, unsigned int*, value )
DEF_FN(nvmlReturn_t, nvmlDeviceGetPcieSpeed, nvmlDevice_t, device, unsigned int*, pcieSpeed )
DEF_FN(nvmlReturn_t, nvmlDeviceGetPcieThroughput, nvmlDevice_t, device, nvmlPcieUtilCounter_t, counter, unsigned int*, value )
DEF_FN(nvmlReturn_t, nvmlDeviceGetPerformanceState, nvmlDevice_t, device, nvmlPstates_t*, pState )
DEF_FN(nvmlReturn_t, nvmlDeviceGetPersistenceMode, nvmlDevice_t, device, nvmlEnableState_t*, mode )
DEF_FN(nvmlReturn_t, nvmlDeviceGetPowerManagementDefaultLimit, nvmlDevice_t, device, unsigned int*, defaultLimit )
DEF_FN(nvmlReturn_t, nvmlDeviceGetPowerManagementLimit, nvmlDevice_t, device, unsigned int*, limit )
DEF_FN(nvmlReturn_t, nvmlDeviceGetPowerManagementLimitConstraints, nvmlDevice_t, device, unsigned int*, minLimit, unsigned int*, maxLimit )
DEF_FN(nvmlReturn_t, nvmlDeviceGetPowerManagementMode, nvmlDevice_t, device, nvmlEnableState_t*, mode )
#if NVML_API_VERSION >= 12
DEF_FN(nvmlReturn_t, nvmlDeviceGetPowerSource, nvmlDevice_t, device, nvmlPowerSource_t*, powerSource )
#endif
DEF_FN(nvmlReturn_t, nvmlDeviceGetPowerState, nvmlDevice_t, device, nvmlPstates_t*, pState )
DEF_FN(nvmlReturn_t, nvmlDeviceGetPowerUsage, nvmlDevice_t, device, unsigned int*, power )
DEF_FN(nvmlReturn_t, nvmlDeviceGetRemappedRows, nvmlDevice_t, device, unsigned int*, corrRows, unsigned int*, uncRows, unsigned int*, isPending, unsigned int*, failureOccurred )
DEF_FN(nvmlReturn_t, nvmlDeviceGetRetiredPages, nvmlDevice_t, device, nvmlPageRetirementCause_t, cause, unsigned int*, pageCount, unsigned long long*, addresses )
DEF_FN(nvmlReturn_t, nvmlDeviceGetRetiredPagesPendingStatus, nvmlDevice_t, device, nvmlEnableState_t*, isPending )
DEF_FN(nvmlReturn_t, nvmlDeviceGetRetiredPages_v2, nvmlDevice_t, device, nvmlPageRetirementCause_t, cause, unsigned int*, pageCount, unsigned long long*, addresses, unsigned long long*, timestamps )
#if NVML_API_VERSION >= 12
DEF_FN(nvmlReturn_t, nvmlDeviceGetRowRemapperHistogram, nvmlDevice_t, device, nvmlRowRemapperHistogramValues_t*, values )
#endif
DEF_FN(nvmlReturn_t, nvmlDeviceGetSamples, nvmlDevice_t, device, nvmlSamplingType_t, type, unsigned long long, lastSeenTimeStamp, nvmlValueType_t*, sampleValType, unsigned int*, sampleCount, nvmlSample_t*, samples )
DEF_FN(nvmlReturn_t, nvmlDeviceGetSerial, nvmlDevice_t, device, char*, serial, unsigned int,  length )
DEF_FN(nvmlReturn_t, nvmlDeviceGetSupportedClocksThrottleReasons, nvmlDevice_t, device, unsigned long long*, supportedClocksThrottleReasons )
DEF_FN(nvmlReturn_t, nvmlDeviceGetSupportedGraphicsClocks, nvmlDevice_t, device, unsigned int,  memoryClockMHz, unsigned int*, count, unsigned int*, clocksMHz )
DEF_FN(nvmlReturn_t, nvmlDeviceGetSupportedMemoryClocks, nvmlDevice_t, device, unsigned int*, count, unsigned int*, clocksMHz )
DEF_FN(nvmlReturn_t, nvmlDeviceGetTargetFanSpeed, nvmlDevice_t, device, unsigned int,  fan, unsigned int*, targetSpeed )
DEF_FN(nvmlReturn_t, nvmlDeviceGetTemperature, nvmlDevice_t, device, nvmlTemperatureSensors_t, sensorType, unsigned int*, temp )
DEF_FN(nvmlReturn_t, nvmlDeviceGetTemperatureThreshold, nvmlDevice_t, device, nvmlTemperatureThresholds_t, thresholdType, unsigned int*, temp )
#if NVML_API_VERSION >= 12
DEF_FN(nvmlReturn_t, nvmlDeviceGetThermalSettings, nvmlDevice_t, device, unsigned int,  sensorIndex, nvmlGpuThermalSettings_t*, pThermalSettings )
#endif
DEF_FN(nvmlReturn_t, nvmlDeviceGetTopologyCommonAncestor, nvmlDevice_t, device1, nvmlDevice_t, device2, nvmlGpuTopologyLevel_t*, pathInfo )
DEF_FN(nvmlReturn_t, nvmlDeviceGetTopologyNearestGpus, nvmlDevice_t, device, nvmlGpuTopologyLevel_t, level, unsigned int*, count, nvmlDevice_t*, deviceArray )
DEF_FN(nvmlReturn_t, nvmlDeviceGetTotalEccErrors, nvmlDevice_t, device, nvmlMemoryErrorType_t, errorType, nvmlEccCounterType_t, counterType, unsigned long long*, eccCounts )
DEF_FN(nvmlReturn_t, nvmlDeviceGetTotalEnergyConsumption, nvmlDevice_t, device, unsigned long long*, energy )
DEF_FN(nvmlReturn_t, nvmlDeviceGetUUID, nvmlDevice_t, device, char*, uuid, unsigned int,  length )
DEF_FN(nvmlReturn_t, nvmlDeviceGetUtilizationRates, nvmlDevice_t, device, nvmlUtilization_t*, utilization )
DEF_FN(nvmlReturn_t, nvmlDeviceGetVbiosVersion, nvmlDevice_t, device, char*, version, unsigned int,  length )
DEF_FN(nvmlReturn_t, nvmlDeviceGetViolationStatus, nvmlDevice_t, device, nvmlPerfPolicyType_t, perfPolicyType, nvmlViolationTime_t*, violTime )
DEF_FN(nvmlReturn_t, nvmlDeviceOnSameBoard, nvmlDevice_t, device1, nvmlDevice_t, device2, int*, onSameBoard )
DEF_FN(nvmlReturn_t, nvmlDeviceResetApplicationsClocks, nvmlDevice_t, device )
DEF_FN(nvmlReturn_t, nvmlDeviceSetAutoBoostedClocksEnabled, nvmlDevice_t, device, nvmlEnableState_t, enabled )
DEF_FN(nvmlReturn_t, nvmlDeviceSetDefaultAutoBoostedClocksEnabled, nvmlDevice_t, device, nvmlEnableState_t, enabled, unsigned int,  flags )
DEF_FN(nvmlReturn_t, nvmlDeviceSetDefaultFanSpeed_v2, nvmlDevice_t, device, unsigned int,  fan )
#if NVML_API_VERSION >= 12
DEF_FN(nvmlReturn_t, nvmlDeviceSetFanControlPolicy, nvmlDevice_t, device, unsigned int,  fan, nvmlFanControlPolicy_t, policy )
#endif
DEF_FN(nvmlReturn_t, nvmlDeviceSetTemperatureThreshold, nvmlDevice_t, device, nvmlTemperatureThresholds_t, thresholdType, int*, temp )
DEF_FN(nvmlReturn_t, nvmlDeviceValidateInforom, nvmlDevice_t, device )
DEF_FN(nvmlReturn_t, nvmlSystemGetTopologyGpuSet, unsigned int,  cpuNumber, unsigned int*, count, nvmlDevice_t*, deviceArray )
DEF_FN(nvmlReturn_t, nvmlVgpuInstanceGetMdevUUID, nvmlVgpuInstance_t, vgpuInstance, char*, mdevUuid, unsigned int,  size )

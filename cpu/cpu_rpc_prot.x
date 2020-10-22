/*
 * msg.x: Remote message printing protocol
 */


typedef char rpc_uuid<16>;
typedef opaque mem_data<>;
typedef uint64_t size_t;
typedef uint64_t ptr;

struct ptr_ptr {
    uint64_t ptr1;
    uint64_t ptr2;
};

struct dint {
    int i1;
    int i2;
};

union int_result switch (int err) {
case 0:
    int data;
default:
    void;
};

union dint_result switch (int err) {
case 0:
    struct dint data;
default:
    void;
};

union float_result switch (int err) {
case 0:
    float data;
default:
    void;
};

union d_result switch (int err) {
case 0:
    double data;
default:
    void;
};

union u64_result switch (int err) {
case 0:
    uint64_t u64;
default:
    void;
};

union ptr_result switch (int err) {
case 0:
    uint64_t ptr;
default:
    void;
};

union str_result switch (int err) {
case 0:
    string str<128>;
default:
    void;
};

union uuid_result switch (int err) {
case 0:
    char bytes[16];
default:
    void;
};

union mem_result switch (int err) {
case 0:
    mem_data data;
default:
    void;
};

struct rpc_dim3 {
    unsigned int x;
    unsigned int y;
    unsigned int z;
};

struct rpc_fatCubin {
    uint32_t magic;
    uint32_t seq;
    uint64_t text;
    uint64_t data;
    uint64_t ptr;
    uint64_t ptr2;
    uint64_t zero;
};

program RPC_CD_PROG {
    version RPC_CD_VERS {
        int          rpc_deinit(void)                                     = 0;
        int          PRINTMESSAGE(string)                                 = 1;
/* RUNTIME API */
        /* Device Management */
        int_result   CUDA_CHOOSE_DEVICE(mem_data)                       = 101;
        int_result   CUDA_DEVICE_GET_ATTRIBUTE(int, int)                = 102;
        int_result   CUDA_DEVICE_GET_BY_PCI_BUS_ID(string<>)            = 103;
        int_result   CUDA_DEVICE_GET_CACHE_CONFIG(void)                 = 104;
        u64_result   CUDA_DEVICE_GET_LIMIT(int)                         = 105;
        /*mem_result CUDA_DEVICE_GET_NVSCISYNC_ATTRIBUTES(int ,int)     = 106;*/
        int_result   CUDA_DEVICE_GET_P2P_ATTRIBUTE(int, int, int)       = 107;
        str_result   CUDA_DEVICE_GET_PCI_BUS_ID(int, int)               = 108;
        int_result   CUDA_DEVICE_GET_SHARED_MEM_CONFIG(void)            = 109;
        dint_result  CUDA_DEVICE_GET_STREAM_PRIORITY_RANGE(void)        = 110;
        u64_result   CUDA_DEVICE_GET_TEXTURE_LMW(mem_data, int)         = 111;
        int          CUDA_DEVICE_RESET(void)                            = 112;
        int          CUDA_DEVICE_SET_CACHE_CONFIG(int)                  = 113;
        int          CUDA_DEVICE_SET_LIMIT(int, size_t)                 = 114;
        int          CUDA_DEVICE_SET_SHARED_MEM_CONFIG(int)             = 115;
        int          CUDA_DEVICE_SYNCHRONIZE(void)                      = 116;
        int_result   CUDA_GET_DEVICE(void)                              = 117;
        int_result   CUDA_GET_DEVICE_COUNT(void)                        = 118;
        int_result   CUDA_GET_DEVICE_FLAGS(void)                        = 119;
        mem_result   CUDA_GET_DEVICE_PROPERTIES(int)                    = 120;
        /*int        CUDA_IPC_CLOSE_MEM_HANDLE(ptr)                     = 121;*/
        /*ptr_result CUDA_IPC_GET_EVENT_HANDLE(int)                     = 122;*/
        /*ptr_result CUDA_IPC_GET_MEM_HANDLE(ptr)                       = 123;*/
        /*ptr_result CUDA_IPC_OPEN_EVENT_HANDLE(ptr)                    = 124;*/
        /*ptr_result CUDA_IPC_OPEN_MEM_HANDLE(ptr, int)                 = 125;*/
        int          CUDA_SET_DEVICE(int)                               = 126;
        int          CUDA_SET_DEVICE_FLAGS(int)                         = 127;
        int          CUDA_SET_VALID_DEVICES(mem_data, int)              = 128;
        /* Error Handling */
        str_result   CUDA_GET_ERROR_NAME(int)                           = 140;
        str_result   CUDA_GET_ERROR_STRING(int)                         = 141;
        int          CUDA_GET_LAST_ERROR(void)                          = 142;
        int          CUDA_PEEK_AT_LAST_ERROR(void)                      = 143;
        /* Stream Management */
        int          CUDA_CTX_RESET_PERSISTING_L2CACHE(void)            = 250;
        int          CUDA_STREAM_ADD_CALLBACK(ptr, ptr, mem_data, int)  = 251;
        int          CUDA_STREAM_ATTACH_MEM_ASYNC(ptr, ptr, size_t, int)= 252;
        /*int          CUDA_STREAM_BEGIN_CAPTURE(ptr, int)                = 253;*/
        int          CUDA_STREAM_COPY_ATTRIBUTES(ptr, ptr)              = 254;
        ptr_result   CUDA_STREAM_CREATE(void)                           = 255;
        ptr_result   CUDA_STREAM_CREATE_WITH_FLAGS(int)                 = 256;
        ptr_result   CUDA_STREAM_CREATE_WITH_PRIORITY(int, int)         = 257;
        int          CUDA_STREAM_DESTROY(ptr)                           = 258;
        /*ptr_result   CUDA_STREAM_END_CAPTURE(ptr)                       = 259;*/
        /* ?         CUDA_STREAM_GET_ATTRIBUTE(ptr, int)                = 260;*/
        /* ?         CUDA_STREAM_GET_CAPTURE_INFO(ptr)                  = 261;*/
        int_result   CUDA_STREAM_GET_FLAGS(ptr)                         = 262;
        int_result   CUDA_STREAM_GET_PRIORITY(ptr)                      = 263;
        /* ?         CUDA_STREAM_IS_CAPTURING(ptr)                      = 264;*/
        int          CUDA_STREAM_QUERY(ptr)                             = 265;
        /*int          CUDA_STREAM_SET_ATTRIBUTE(ptr, int, ?)             = 266;*/
        int          CUDA_STREAM_SYNCHRONIZE(ptr)                       = 267;
        int          CUDA_STREAM_WAIT_EVENT(ptr, ptr, int)              = 268;
        int_result   CUDA_THREAD_EXCHANGE_STREAM_CAPTURE_MODE(int)      = 269;
        /* Event Management */
        ptr_result   CUDA_EVENT_CREATE(void)                            = 280;
        ptr_result   CUDA_EVENT_CREATE_WITH_FLAGS(int)                  = 281;
        int          CUDA_EVENT_DESTROY(ptr)                            = 282;
        float_result CUDA_EVENT_ELAPSED_TIME(ptr, ptr)                  = 283;
        int          CUDA_EVENT_QUERY(ptr)                              = 284;
        int          CUDA_EVENT_RECORD(ptr, ptr)                        = 285;
        int          CUDA_EVENT_RECORD_WITH_FLAGS(ptr, ptr, int)        = 286;
        int          CUDA_EVENT_SYNCHRONIZE(ptr)                        = 287;
        /* External Resource Interoperability */
            /* NOT IMPLEMENTED */
        /* Execution Control */
        mem_result   CUDA_FUNC_GET_ATTRIBUTES(ptr)                      = 310;
        int          CUDA_FUNC_SET_ATTRIBUTES(ptr, int, int)            = 311;
        int          CUDA_FUNC_SET_CACHE_CONFIG(ptr, int)               = 312;
        int          CUDA_FUNC_SET_SHARED_MEM_CONFIG(ptr, int)          = 313;
        int          CUDA_LAUNCH_COOPERATIVE_KERNEL(ptr, rpc_dim3, 
                          rpc_dim3, mem_data, size_t, ptr)              = 314;
        int          CUDA_LAUNCH_COOPERATIVE_KERNEL_MULTI_DEVICE(ptr, rpc_dim3,
                          rpc_dim3, mem_data, size_t, ptr, int, int)    = 315;
        /*int          CUDA_LAUNCH_HOST_FUNC(ptr, ptr, mem_data)        = 316;*/
        int          CUDA_LAUNCH_KERNEL(ptr, rpc_dim3, rpc_dim3, mem_data, 
                          size_t, ptr)                                  = 317;
        d_result     CUDA_SET_DOUBLE_FOR_DEVICE(double)                 = 318;
        d_result     CUDA_SET_DOUBLE_FOR_HOST(double)                   = 319;
        /* Occupancy */
        u64_result   CUDA_OCCUPANCY_AVAILABLE_DSMPB(ptr, int, int)      = 330;
        int_result   CUDA_OCCUPANCY_MAX_ACTIVE_BPM(ptr, int, size_t)    = 331;
        int_result   CUDA_OCCUPANCY_MAX_ACTIVE_BPM_WITH_FLAGS(ptr, int,
                          size_t, int)                                  = 332;


        
        /* Others */
        ptr_result   CUDA_MALLOC(size_t)                                  = 2;
        int          CUDA_MEMCPY_HTOD(ptr, mem_data, size_t)              = 3;
        mem_result   CUDA_MEMCPY_DTOH(ptr, size_t)                        = 4;
        int          CUDA_FREE(ptr)                                       = 6;
        int          CUDA_HOST_ALLOC(int, size_t, ptr, unsigned int)      = 20;
        int          CUDA_FREE_HOST(int)                                  = 21;   
        int          CUDA_MEMCPY_SHM(int, ptr, size_t, int)               = 22;
        int          CUDA_MEMCPY_DTOD(ptr, ptr, size_t)                   = 23;
        int          CUDA_MEMCPY_TO_SYMBOL(ptr, mem_data, size_t, size_t) = 27;
        int          CUDA_MEMCPY_TO_SYMBOL_SHM(int, ptr, size_t, size_t, int) = 28;
        int          CUDA_MEMCPY_IB(int, ptr, size_t, int)                = 31;
        /* DRIVER API */
        int_result  rpc_cuDeviceGetCount(void)                          = 1002;
        int         rpc_cuInit(int)                                     = 1003;
        int_result  rpc_cuDriverGetVersion(void)                        = 1004;
        int_result  rpc_cuDeviceGet(int)                                = 1005;
        str_result  rpc_cuDeviceGetName(int)                            = 1006;
        u64_result  rpc_cuDeviceTotalMem(int)                           = 1007;
        int_result  rpc_cuDeviceGetAttribute(int, int)                  = 1008;
        uuid_result rpc_cuDeviceGetUuid(int)                            = 1009;
        ptr_result  rpc_cuCtxGetCurrent(void)                           = 1010;
        int         rpc_cuCtxSetCurrent(uint64_t)                       = 1011;
        ptr_result  rpc_cuDevicePrimaryCtxRetain(int)                   = 1012;
        ptr_result  rpc_cuModuleGetFunction(uint64_t, string)           = 1013;
        ptr_result  rpc_cuGetExportTable(rpc_uuid uuid)                 = 1014;
        ptr_result  rpc_cuMemAlloc(uint64_t)                            = 1015;
        int_result  rpc_cuCtxGetDevice(void)                            = 1016;
        int         rpc_cuMemcpyHtoD(uint64_t, mem_data)                = 1017;
        int         rpc_cuLaunchKernel(uint64_t, unsigned int,
                                       unsigned int, unsigned int,
                                       unsigned int, unsigned int,
                                       unsigned int, unsigned int, 
                                       uint64_t, mem_data)              = 1018;
        /* HIDDEN DRIVER API */
        ptr_result  rpc_hidden_get_device_ctx(int)                      = 1101;
        ptr_result  rpc_hidden_get_module(uint64_t arg2, uint64_t arg3,
                                          uint64_t arg4, int arg5)      = 1105;
        ptr_result  rpc_hidden_1_1(void)                                = 1111;
        void        rpc_hidden_1_3(uint64_t, uint64_t)                  = 1113;
        ptr_result  rpc_hidden_1_5(void)                                = 1115;
        void        rpc_hidden_2_1(uint64_t)                            = 1121;
        int         rpc_hidden_3_0(int, uint64_t, uint64_t)             = 1130;
        mem_result  rpc_hidden_3_2(int, uint64_t)                       = 1132;
        /* CUSOLVER API */
        ptr_result  rpc_cusolverDnCreate(void)                          = 2001;
        int         rpc_cusolverDnSetStream(ptr, ptr)                   = 2002;
        int_result  rpc_cusolverDnDgetrf_bufferSize(ptr, int, int, 
                                                    ptr, int)           = 2003;
        int         rpc_cusolverDnDgetrf(ptr, int, int, ptr, int,
                                         ptr, ptr, ptr)                 = 2004;
        int         rpc_cusolverDnDgetrs(ptr, int, int, int, ptr,
                                         int, ptr, ptr, int, ptr)       = 2005;
        int         rpc_cusolverDnDestroy(ptr)                          = 2006;
        
    } = 1;
} = 99;


/*
 * msg.x: Remote message printing protocol
 */


typedef char rpc_uuid<16>;
typedef opaque mem_data<>;
typedef uint64_t size_t;
typedef uint64_t ptr;

union int_result switch (int err) {
case 0:
    int data;
default:
    void;
};

union float_result switch (int err) {
case 0:
    float data;
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

struct ptr_ptr {
    uint64_t ptr1;
    uint64_t ptr2;
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
        int          PRINTMESSAGE(string)                                 = 1;
/* RUNTIME API */
        ptr_result   CUDA_MALLOC(size_t)                                  = 2;
        int          CUDA_MEMCPY_HTOD(ptr, mem_data, size_t)              = 3;
        mem_result   CUDA_MEMCPY_DTOH(ptr, size_t)                        = 4;
        int          CUDA_LAUNCH_KERNEL(ptr, rpc_dim3, rpc_dim3, mem_data, size_t, ptr) = 5;
        int          CUDA_FREE(ptr)                                       = 6;
        int          CUDA_DEVICE_SYNCHRONIZE(void)                        = 7;
        int_result   CUDA_GET_DEVICE_COUNT(void)                          = 8;
        int_result   CUDA_DEVICE_GET_ATTRIBUTE(int, int)                  = 9;
        int          CUDA_SET_DEVICE(int)                                 = 10;
        ptr_result   CUDA_EVENT_CREATE(void)                              = 11;
        ptr_result   CUDA_STREAM_CREATE_WITH_FLAGS(int)                   = 12;
        int          CUDA_STREAM_SYNCHRONIZE(ptr)                         = 13;
        int          CUDA_EVENT_RECORD(ptr, ptr)                          = 14;
        float_result CUDA_EVENT_ELAPSED_TIME(ptr, ptr)                    = 15;
        int          CUDA_EVENT_DESTROY(ptr)                              = 16;
        int          CUDA_EVENT_SYNCHRONIZE(ptr)                          = 17;
        mem_result   CUDA_GET_DEVICE_PROPERTIES(int)                      = 18;
        str_result   CUDA_GET_ERROR_NAME(int)                             = 19;
        int          CUDA_HOST_ALLOC(int, size_t, ptr)                    = 20;
        int          CUDA_FREE_HOST(ptr)                                  = 21;   
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
        
    } = 1;
} = 99;


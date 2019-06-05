/*
 * msg.x: Remote message printing protocol
 */

union int_result switch (int err) {
case 0:
    int data;
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

typedef char rpc_uuid<16>;

program RPC_CD_PROG {
    version RPC_CD_VERS {
        int PRINTMESSAGE(string)                                        = 1;
        int_result  rpc_cuDeviceGetCount(void)                          = 2;
        int         rpc_cuInit(int)                                     = 3;
        int_result  rpc_cuDriverGetVersion(void)                        = 4;
        int_result  rpc_cuDeviceGet(int)                                = 5;
        str_result  rpc_cuDeviceGetName(int)                            = 6;
        u64_result  rpc_cuDeviceTotalMem(int)                           = 7;
        int_result  rpc_cuDeviceGetAttribute(int, int)                  = 8;
        uuid_result rpc_cuDeviceGetUuid(int)                            = 9;
        ptr_result  rpc_cuCtxGetCurrent(void)                           = 10;
        int         rpc_cuCtxSetCurrent(uint64_t)                       = 11;
        ptr_result  rpc_cuDevicePrimaryCtxRetain(int)                   = 12;
        ptr_result  rpc_cuModuleGetFunction(uint64_t, string)           = 13;
        ptr_result  rpc_cuGetExportTable(rpc_uuid uuid)                 = 14;
        ptr_result  rpc_cuMemAlloc(uint64_t)                            = 15;
        int_result  rpc_cuCtxGetDevice(void)                            = 16;
        
        ptr_result  rpc_hidden_get_device_ctx(int)                      = 101;
        ptr_result  rpc_hidden_get_module(uint64_t arg2, uint64_t arg3, uint64_t arg4, int arg5)                                                     = 105;
        ptr_result  rpc_hidden_1_1(void)                                = 111;
        void        rpc_hidden_1_3(uint64_t, uint64_t)                  = 113;
        ptr_result  rpc_hidden_1_5(void)                                = 115;
        void        rpc_hidden_2_1(uint64_t)                            = 121;
        int         rpc_hidden_3_0(int, uint64_t, uint64_t)             = 130;
        ptr_result  rpc_hidden_3_2(int, uint64_t)                       = 132;

    } = 1;
} = 99;

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

union str_result switch (int err) {
case 0:
    string str<>;
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
        int_result  rpc_cuGetExportTable(rpc_uuid uuid)                 = 14;
        
        ptr_result  rpc_hidden_get_device_ctx(int)                      = 101;

    } = 1;
} = 99;

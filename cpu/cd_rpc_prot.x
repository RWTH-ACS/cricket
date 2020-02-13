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
        int         PRINTMESSAGE(string)                                 = 1;
        ptr_result  CUDA_MALLOC(size_t)                                  = 2;
        int         CUDA_MEMCPY_HTOD(ptr, mem_data, size_t)              = 3;
        mem_result  CUDA_MEMCPY_DTOH(ptr, size_t)                        = 4;
        int         CUDA_LAUNCH_KERNEL(ptr, rpc_dim3, rpc_dim3, mem_data, size_t, ptr) = 5;
        int         CUDA_FREE(ptr)                                       = 6;
    } = 1;
} = 99;


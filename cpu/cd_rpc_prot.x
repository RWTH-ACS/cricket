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
typedef opaque mem_data<>;

program RPC_CD_PROG {
    version RPC_CD_VERS {
        int PRINTMESSAGE(string)                                        = 1;
    } = 1;
} = 99;


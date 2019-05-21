/*
 * msg.x: Remote message printing protocol
 */
union int_result switch (int err) {
case 0:
    int data;
default:
    void;
};

program RPC_CD_PROG {
    version RPC_CD_VERS {
        int PRINTMESSAGE(string) = 1;
        int_result CuDeviceGetCount(void) = 2;
    } = 1;
} = 99;

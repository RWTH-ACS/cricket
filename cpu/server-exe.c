#include "cpu_rpc_prot.h"
#include "cpu-server.h"
#include "log.h"

#include <stdlib.h>
#include <stdint.h>

int main(int argc, char** argv)
{
    if (argc == 1) {
        cricket_main(RPC_CD_PROG, RPC_CD_VERS);
    } else if (argc == 2) {
        uint64_t vers;
        if (sscanf(argv[1], "%lu", &vers) != 1) {
            printf("version string could not be converted to number\n");
            printf("usage: %s [unique rpc version]\n", argv[0]);
            return 1;
        }
        cricket_main(RPC_CD_PROG, vers);
    } else {
        printf("usage: %s\n", argv[0]);
    }
    return 0;
}

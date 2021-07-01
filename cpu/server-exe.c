#include "cpu_rpc_prot.h"
#include "cpu-server.h"
#include "log.h"

#include <stdlib.h>

int main(int argc, char** argv)
{

    //TODO: Check if command path exists
    if (argc == 1) {
        cricket_main_static(RPC_CD_PROG, RPC_CD_VERS);
    } else if (argc == 2) {
        cricket_main_hash(argv[1]);
    } else {
        LOGE(LOG_ERROR, "usage: %s [command]", argv[0]);
    }
    return 0;
}

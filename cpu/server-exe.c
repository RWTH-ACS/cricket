
#include "cpu-server.h"
#include "log.h"

#include <stdlib.h>
#include <dlfcn.h>

int main(int argc, char** argv)
{

    //TODO: Check if command path exists
    if (argc == 1) {
        cricket_main("/proc/self/exe");
    } else if (argc == 2) {
        cricket_main(argv[1]);
    } else {
        LOGE(LOG_ERROR, "usage: %s [command]", argv[0]);
    }
    return 0;
}


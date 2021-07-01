
#include "cpu-server.h"
#include "log.h"

/* shared object constructor; executes before main and thus hijacks main program */
void __attribute__ ((constructor)) library_constr(void)
{
    cricket_main_hash("/proc/self/exe");
}


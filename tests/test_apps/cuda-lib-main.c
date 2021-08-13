#include <stdlib.h>
#include <stdio.h>
#include <dlfcn.h>

int main()
{
    void *dlhandle;

    if ((dlhandle = dlopen("./cuda-lib.so", RTLD_LAZY)) == NULL) {
        printf("error opening library\n");
        return 1;
    }

    void (*fn)(void);
    if ((fn = dlsym(dlhandle, "call_kernel")) == NULL) {
        printf("dlsym failed\n");
        return 1;
    }


    fn();

    dlclose(dlhandle);
    return 0;
}

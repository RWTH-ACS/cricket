#include <dlfcn.h>
#include <stdio.h>

int main(int argc, char** argv)
{
    void *dlhandle;

    if ((dlhandle = dlopen("./libtest_kernel.so", RTLD_LAZY)) == NULL) {
        printf("error opening library\n");
        return 1;
    }

    int (*fn)(void);

    printf("kernel: %p\n", dlsym(dlhandle, "_Z6kernelPtS_S_csix"));

    if ((fn = dlsym(dlhandle, "main")) == NULL) {
        printf("dlsym failed\n");
        return 1;
    }

    fn();

    return 0;
}


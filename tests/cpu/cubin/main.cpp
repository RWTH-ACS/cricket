#include <stdio.h>

#include <cuda_runtime.h>
#include <cuda.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>


#define printCudaErrors(err) __printCudaErrors (err, __FILE__, __LINE__)
#define printRtErrors(err) __printRtErrors (err, __FILE__, __LINE__)

inline void __printCudaErrors(CUresult err, const char *file, const int line )
{
    const char* err_str;
    if (err != CUDA_SUCCESS) {
        cuGetErrorString(err, &err_str);
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
                file, line, (int)err, err_str );
        exit(1);
    }
}

inline void __printRtErrors(cudaError_t err, const char *file, const int line )
{
    if (err != cudaSuccess) {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
                file, line, (int)err, cudaGetErrorString(err));
        exit(1);
    }
}

void prepare_mem(int **mem, size_t len)
{
    cudaError_t rterr;
    int host_mem[len];
    if ((rterr = cudaMalloc(mem, len*sizeof(int))) != cudaSuccess) {
        printRtErrors(rterr);
    }

    for (size_t i = 0; i < len; i++) {
        host_mem[i] = (int)i;
    }

    if ((rterr = cudaMemcpy(*mem, host_mem, len*sizeof(int), cudaMemcpyHostToDevice)) != cudaSuccess) {
        printRtErrors(rterr);
    }
}

void check_free_mem(int *mem, size_t len)
{
    cudaError_t rterr;
    int host_mem[len];

    if ((rterr = cudaMemcpy(host_mem, mem, len*sizeof(int), cudaMemcpyDeviceToHost)) != cudaSuccess) {
        printRtErrors(rterr);
    }

    bool ok = true;
    for (size_t i = 0; i < len; i++) {
        if (host_mem[i] != (int)(i+1)) {
            printf("err at %d\n", (int)i);
            ok = false;
        }
    }
    if (ok) printf("successful!\n");

    cudaFree(mem);
}

int getModuleFromCubin(CUmodule *module, const char *cubin)
{
    CUresult err;
    if ((err = cuModuleLoad(module, "kernel.cubin")) != CUDA_SUCCESS) {
        printCudaErrors(err);
        return 1;
    }
    return 0;
}

int getModuleFromCubinInMemory(CUmodule *module, const char *cubin)
{
    int fd = open(cubin, O_RDONLY);
    if (fd < 0) {
        printf("error\n");
        return 1;
    }
    struct stat st;
    if (fstat(fd, &st) < 0) {
        printf("error\n");
        return 1;
    }
    printf("size: %#0zx\n", (int)st.st_size);
    void *buf = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (buf == MAP_FAILED) {
        printf("error\n");
        return 1;
    }
    CUresult err;
    if ((err = cuModuleLoadData(module, buf)) != CUDA_SUCCESS) {
        printCudaErrors(err);
        return 1;
    }
    return 0;
}

int getModuleFromShared(CUmodule **module, const char *cubin)
{
    return 0;
}

int main(int argc, char** argv)
{
    CUresult err;
    cudaError_t rterr;

    if ((rterr = cudaSetDevice(0)) != cudaSuccess) {
        printRtErrors(rterr);
    }

    int *mem;
    size_t len = 32;
    prepare_mem(&mem, len);

    CUmodule module;
    CUfunction func;
    printf("testing cubin...\n");
    if (getModuleFromCubinInMemory(&module, "kernel.cubin") != 0) {
        printf("error\n");
        return 1;
    }
    // if (getModuleFromCubin(&module, "kernel.cubin") != 0) {
    //     printf("error\n");
    //     return 1;
    // }
    // if ((err = getModuleFromShared(&module, "kernel.so")) != 0) {
    //     printf("error\n");
    //     return 1;
    // }

    if ((err = cuModuleGetFunction(&func, module, "kernel")) != CUDA_SUCCESS) {
        printCudaErrors(err);
    }
    sleep(5);
    int a = 4;
    void *params[] = {&a, &mem, &len};
    if ((err = cuLaunchKernel(func, 1, 1, 1, len, 1, 1, 8, CU_STREAM_PER_THREAD, params, NULL)) != CUDA_SUCCESS) {
        printCudaErrors(err);
    }
    check_free_mem(mem, len);
    cuModuleUnload(module);

  /*  prepare_mem(&mem, len);
    printf("testing fatbin...\n");
    if ((err = cuModuleLoad(&module, "kernel.fatbin")) != CUDA_SUCCESS) {
        printCudaErrors(err);
    }

    if ((err = cuModuleGetFunction(&func, module, "kernel")) != CUDA_SUCCESS) {
        printCudaErrors(err);
    }

    params[1] = &mem;
    if ((err = cuLaunchKernel(func, 1, 1, 1, len, 1, 1, 8, CU_STREAM_PER_THREAD, params, NULL)) != CUDA_SUCCESS) {
        printCudaErrors(err);
    }
    check_free_mem(mem, len);*/
}

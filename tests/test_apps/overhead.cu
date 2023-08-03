#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

#include <cuda_runtime.h>

#define ITERATIONS 100000
#define WARMUPS 100
#define MEMSIZE 1024*1024
const int blocksize = 32;

__global__
void kernel(uint16_t *A, uint16_t *x, uint16_t *res, char b, short c, int a, long long int d)
{
    int i = threadIdx.x;
    res[i] = A[i] * x[i];
}

__global__
void kernel_no_param(void)
{
}

int main()
{
    struct timeval begin, end;

    int iterations = ITERATIONS;
    int cnt;

    printf("init CUDA\n");
    cudaGetDeviceCount(&cnt);
    printf("1. cudaGetDeviceCount (%d iterations)\n", iterations);
    for (int i=0; i != WARMUPS; i++) {
        cudaGetDeviceCount(&cnt);
    }
    gettimeofday(&begin, NULL);
    for (int i=0; i != iterations; i++) {
        cudaGetDeviceCount(&cnt);
    }
    gettimeofday(&end, NULL);
    printf("TOTALTIME: %0u.%06u\n\n", (end.tv_sec - begin.tv_sec), (end.tv_usec - begin.tv_usec));

    uint16_t *dev_A;
    size_t A_size = MEMSIZE;
    printf("2. cudaMalloc/cudaFree (%d iterations)\n", iterations);
    for (int i=0; i != WARMUPS; i++) {
        cudaMalloc( (void**)&dev_A, A_size );
        cudaFree( dev_A );
    }
    gettimeofday(&begin, NULL);
    for (int i=0; i != iterations; i++) {
        cudaMalloc( (void**)&dev_A, A_size );
        cudaFree( dev_A );
    }
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    printf("TOTALTIME: %0u.%06u\n\n", (end.tv_sec - begin.tv_sec), (end.tv_usec - begin.tv_usec));

    dim3 dimBlock( blocksize, 1 );
    dim3 dimGrid( 1, 1);
    printf("3. kernel launch w/o parameteter (%d iterations)\n", iterations);
    for (int i=0; i != WARMUPS; i++) {
        kernel_no_param<<<dimGrid, dimBlock>>>();
    }
    gettimeofday(&begin, NULL);
    for (int i=0; i != iterations; i++) {
        kernel_no_param<<<dimGrid, dimBlock>>>();
    }
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    cudaError_t result = cudaGetLastError();
    printf("\nresult: %d\n", result);
    printf("TOTALTIME: %0u.%06u\n\n", (end.tv_sec - begin.tv_sec), (end.tv_usec - begin.tv_usec));

    uint16_t *dev_x;
    uint16_t *dev_res;
    cudaMalloc( (void**)&dev_x, MEMSIZE );
    cudaMalloc( (void**)&dev_res, sizeof(int) );
    cudaMalloc( (void**)&dev_A, MEMSIZE );
    cudaMemset( dev_A, 1, MEMSIZE);
    cudaMemset( dev_x, 2, MEMSIZE);
    printf("4. kernel launch w/ parameteters (%d iterations)\n", iterations);
    for (int i=0; i != WARMUPS; i++) {
        kernel<<<dimGrid, dimBlock>>>(dev_A, dev_x, dev_res, 0, 0, 0, 0);
    }
    gettimeofday(&begin, NULL);
    for (int i=0; i != iterations; i++) {
        kernel<<<dimGrid, dimBlock>>>(dev_A, dev_x, dev_res, 0, 0, 0, 0);
    }
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    result = cudaGetLastError();
    printf("\nresult: %d\n", result);
    printf("TOTALTIME: %0u.%06u\n\n", (end.tv_sec - begin.tv_sec), (end.tv_usec - begin.tv_usec));

    return 0;
}

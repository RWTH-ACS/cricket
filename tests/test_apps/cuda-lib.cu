#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

#include <cuda.h>

__global__
void hello_world(void)
{
    printf("hello world\n");
}


extern "C" int call_kernel(void)
{
    int cnt;

    cudaGetDeviceCount(&cnt);
    printf("got %d devices\n", cnt);

    //cudaMalloc( (void**)&dev_x, x_size );
    //cudaMemcpy( dev_x, x, x_size, cudaMemcpyHostToDevice );


    dim3 dimBlock( 1, 1 );
    dim3 dimGrid( 1, 1);
    hello_world<<<dimGrid, dimBlock>>>();

    //cudaMemcpy( res, dev_res, x_size, cudaMemcpyDeviceToHost );
    //cudaFree( dev_A );
    printf("have a nice day\n");
    return 0;
}

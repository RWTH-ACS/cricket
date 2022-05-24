
#include <stdio.h>
//disable mangling
extern "C" {

__global__
void kernel(int a, int *mem, size_t len)
{
    printf("i am working. got param %d\n", a);
    if (threadIdx.x < len) {
        mem[threadIdx.x] += 1;
    }
}


}

int main()
{
    printf("hello world!\n");
}


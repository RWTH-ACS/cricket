#ifndef _CRICKET_CUDA_H_
#define _CRICKET_CUDA_H_

/*__device__
inline void CRICKET_CR_ENABLE(const int N) 
{
    const uint32_t n_const = 0;
    asm volatile(".reg  .u32 n;  \
                  ld.const.u32 n, [%0]; \
                  .reg .pred p; \
                  setp.eq.u32  p, n,0; \
                  @p bra L1;" : : "r"(n_const)); \
    #pragma unroll \
    for (int i=0; i < N; ++i) {asm volatile("brkpt;");} \
    asm volatile ("L1:");
}*/

__device__
inline void CRICKET_CR_ENABLE(const int N) 
{
    volatile static uint32_t n_const = 0;
    if (n_const != 0) {
        #pragma unroll 
        for (int i=0; i < N; ++i) {asm volatile("brkpt;");} 
    }
}


#endif

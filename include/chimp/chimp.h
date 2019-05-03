#ifndef _CHIMP_H_
#define _CHIMP_H_

typedef struct chimp_libc_ops
{
    void *(*calloc)(size_t nmemb, size_t size);
    void *(*malloc)(size_t size);
    void (*free)(void *ptr);
    void *(*realloc)(void *ptr, size_t size);
    void *(*memalign)(size_t blocksize, size_t bytes);
} chimp_libc_ops_t;

#endif //_CHIMP_H_

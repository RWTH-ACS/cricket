/* CHIMP - Cricket Helper Injected into Memory Procedures
 *
 * Based on FatalFlaw:
 * https://stackoverflow.com/questions/6083337/overriding-malloc-using-the-ld-preload-mechanism
 **/

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include <pthread.h>

#include "libchimp.h"
#include "chimp.h"
#include "chimp_list.h"

char tmpbuff[1024];
unsigned long tmppos = 0;
unsigned long tmpallocs = 0;
char log_sw = 0;
// We dont want to log memory operations by chimp so deactivate logging
// while inside chimp functions
char rec_sw = 1;

void *memset(void *,int,size_t);
void *memmove(void *to, const void *from, size_t size);

static chimp_libc_ops_t ops;
static chimp_list_t list;

// TODO: make to list
static pthread_t threads[16];
static int threadcnt = 0;

/*=========================================================
 * interception points
 */

// TODO: Move to other file
// is a critical section!
void add_thread()
{
    pthread_t tid = pthread_self();
    for (int i=0; i < threadcnt; ++i) {
        if (pthread_equal(threads[i], tid)) {
            return;
        }
    }
    printf("new thread (%d).\n", threadcnt+1);
    threads[threadcnt++] = tid;
    pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);
    pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, NULL);
}

void kill_threads()
{
    pthread_t main_thread = pthread_self();
    printf("cancelling %d threads...\n", threadcnt);
    for (int i=0; i < threadcnt; ++i) {
        if (!pthread_equal(main_thread, threads[i])) {
            pthread_cancel(threads[i]);
        }
    }
    printf("joining...\n");
    for (int i=0; i < threadcnt; ++i) {
        if (!pthread_equal(main_thread, threads[i])) {
            pthread_join(threads[i], NULL);
        }
    }
}
// ****

static void init()
{
    ops.malloc     = dlsym(RTLD_NEXT, "malloc");
    ops.free       = dlsym(RTLD_NEXT, "free");
    ops.calloc     = dlsym(RTLD_NEXT, "calloc");
    ops.realloc    = dlsym(RTLD_NEXT, "realloc");
    ops.memalign   = dlsym(RTLD_NEXT, "memalign");

    if (!ops.malloc || !ops.free || !ops.calloc || !ops.realloc ||
        !ops.memalign) {
        fprintf(stderr, "Error in `dlsym`: %s\n", dlerror());
        exit(1);
    }

    if (!chimp_list_init(&list, &ops)) {
        fprintf(stderr, "error while initializing chimp_list\n");
        exit(1);
    }
}

/* Print list of logged memory operations
 * this function is exported
 **/
void chimp_print_list()
{
    int i;
    char *func_str[] = {"malloc", "free"};
    chimp_list_elem_t elem;

    chimp_list_compress(&list);
    for (i = 0; i < list.size; ++i) {
        elem = list.arr[i];
        if (elem.func == FUNC_MALLOC) {
            printf("\t%s(%lu) = %p\n", func_str[elem.func], elem.mem_size,
                   elem.ptr);
        } else if (elem.func == FUNC_FREE) {
            printf("\t%s(%p)\n", func_str[elem.func], elem.ptr);
        }
    }
}

void chimp_free_all()
{
    int i;
    chimp_list_elem_t elem;
    kill_threads();
    pthread_mutex_lock(&list.lock);
    chimp_list_compress(&list);
    for (i=0; i < list.size; ++i) {
        elem = list.arr[i];
        if (elem.func == FUNC_MALLOC) {
            ops.free(elem.ptr);
        }
    }
    pthread_mutex_unlock(&list.lock);
}

/* Turn on/off logging of memory operations
 * this function is exported
 **/
void chimp_malloc_togglelog()
{
    log_sw = !log_sw;
}

void *malloc(size_t size)
{
    static int initializing = 0;
    if (ops.malloc == NULL) {
        if (!initializing) {
            initializing = 1;
            init();
            initializing = 0;
            fprintf(stdout, "malloc: allocated %lu bytes of temp memory in %lu "
                            "chunks during initialization\n",
                    tmppos, tmpallocs);
        } else {
            if (tmppos + size < sizeof(tmpbuff)) {
                void *retptr = tmpbuff + tmppos;
                tmppos += size;
                ++tmpallocs;
                return retptr;
            } else {
                fprintf(stdout, "malloc: too much memory requested during "
                                "initialisation - increase tmpbuff size\n");
                exit(1);
            }
        }
    }
    void *ptr = ops.malloc(size);
    if (log_sw && rec_sw && ptr && size) {
        rec_sw = 0;
        add_thread();
        if (!chimp_list_add(&list, FUNC_MALLOC, ptr, size)) {
            fprintf(stderr, "failed to add malloc(%lu) = %p to list\n", size,
                    ptr);
        }
        rec_sw = 1;
    }
    return ptr;
}

void free(void *ptr)
{
    // something wrong if we call free before one of the allocators!
    //  if (myfn_malloc == NULL)
    //      init();

    if (ptr >= (void *) tmpbuff && ptr <= (void *)(tmpbuff + tmppos))
        fprintf(stdout, "freeing temp memory\n");
    else
        ops.free(ptr);

    if (log_sw && rec_sw && ptr) {
        rec_sw = 0;
        add_thread();
        if (!chimp_list_add(&list, FUNC_FREE, ptr, 0)) {
            fprintf(stderr, "failed to add free(%lu) to list\n", ptr);
        }
        rec_sw = 1;
    }
}

void *realloc(void *ptr, size_t size)
{
    if (ops.malloc == NULL) {
        void *nptr = malloc(size);
        if (nptr && ptr) {
            memmove(nptr, ptr, size);
            free(ptr);
        }
        return nptr;
    }

    void *nptr = ops.realloc(ptr, size);

    if (log_sw && rec_sw && nptr && ptr && size) {
        rec_sw = 0;
        add_thread();
        if (!chimp_list_add(&list, FUNC_FREE, ptr, 0)) {
            fprintf(stderr, "realloc: failed to add free(%p) to list\n",
                    ptr);
        }
        if (!chimp_list_add(&list, FUNC_MALLOC, nptr, size)) {
            fprintf(stderr, "realloc: failed to add malloc(%lu) = %p to list\n",
                    size, nptr);
        }
        rec_sw = 1;
    }
    return nptr;
}

void *calloc(size_t nmemb, size_t size)
{
    if (ops.malloc == NULL) {
        void *ptr = malloc(nmemb * size);
        if (ptr)
            memset(ptr, 0, nmemb * size);
        return ptr;
    }

    void *ptr = ops.calloc(nmemb, size);
    if (log_sw && rec_sw && ptr && size && nmemb) {
        rec_sw = 0;
        add_thread();
        if (!chimp_list_add(&list, FUNC_MALLOC, ptr, nmemb * size)) {
            fprintf(stderr, "calloc: failed to add malloc(%lu) = %p to list\n",
                    nmemb * size, ptr);
        }
        rec_sw = 1;
    }
    return ptr;
}

void *memalign(size_t blocksize, size_t bytes)
{
    void *ptr = ops.memalign(blocksize, bytes);
    if (log_sw && rec_sw && ptr && bytes) {
        rec_sw = 0;
        add_thread();
        if (!chimp_list_add(&list, FUNC_MALLOC, ptr, bytes)) {
            fprintf(stderr, "memalign: failed to add malloc(%lu) = %p to list\n",
                    bytes, ptr);
        }
        rec_sw = 1;
    }
    return ptr;
}

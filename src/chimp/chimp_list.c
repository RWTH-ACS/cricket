
#include <stdio.h>
#include <stddef.h>
#include "chimp_list.h"


bool chimp_list_init(chimp_list_t *list, chimp_libc_ops_t *ops)
{
    if (!(list->arr =
              ops->malloc(CHIMP_LIST_INIT_SIZE * sizeof(chimp_list_elem_t))) ) {
        fprintf(stderr, "chimp_list: malloc failed!\n");
        return false;
    }

    list->alloc_size = CHIMP_LIST_INIT_SIZE;
    list->size = 0;
    list->ops = ops;
    pthread_mutex_init(&list->lock, NULL);
    return true;
}

void chimp_list_free(chimp_list_t *list)
{
    pthread_mutex_lock(&list->lock);
    list->ops->free(list->arr);
    list->alloc_size = 0;
    list->size = 0;
}

bool chimp_list_add_elem(chimp_list_t *list, chimp_list_elem_t elem)
{
    pthread_mutex_lock(&list->lock);
    if (list->alloc_size <= list->size) {
        if (!(list->arr = list->ops->realloc(list->arr, 
                      (list->alloc_size + CHIMP_LIST_INIT_SIZE) * 
                      sizeof(chimp_list_elem_t))) ) {
            fprintf(stderr, "chimp_list: realloc failed!\n");
            return false;
        }
        list->alloc_size += CHIMP_LIST_INIT_SIZE;
    }
    list->arr[list->size++] = elem;
    pthread_mutex_unlock(&list->lock);
    return true;
}

bool chimp_list_add(chimp_list_t *list, enum chimp_list_func func, void *ptr,
                    size_t mem_size)
{
    chimp_list_elem_t elem = {
        .func = func,
        .ptr = ptr,
        .mem_size = mem_size,
    };
    return chimp_list_add_elem(list, elem);
}

/* Search for frees and their corresponding mallocs and delete both.
 * We are only interested in still allocated memory.
 **/
static bool compress_freed(chimp_list_t *list)
{
    size_t free_i;
    size_t malloc_i;
    size_t next_free_i = list->size;

    // Start by finding the last free
    while (next_free_i > 0) {
        --next_free_i;
        if (list->arr[next_free_i].func == FUNC_FREE && 
            list->arr[next_free_i].ptr != NULL) {
            break;
        }
    }
    
    // If we didn't find a free, this loop will not run. A free at position 0
    // can never be deleted, because earlier mallocs are not recorded.
    while (next_free_i > 0) {
        free_i = next_free_i;
        malloc_i = next_free_i;
        next_free_i = 0;
        while (malloc_i > 0 ) {
            malloc_i--;
            if (list->arr[malloc_i].func == FUNC_MALLOC &&
                list->arr[malloc_i].ptr == list->arr[free_i].ptr) {

                list->arr[malloc_i].ptr = NULL;
                list->arr[free_i].ptr = NULL;
                break;
            } else if (next_free_i == 0 &&
                       list->arr[malloc_i].func == FUNC_FREE && 
                       list->arr[malloc_i].ptr != NULL) {
                // While we are searching for mallocs, we can also look for
                // frees
                next_free_i = malloc_i;
            }
        }

        // Did we not find the corresponding malloc? Then do a detailed search
        if (malloc_i == 0 && list->arr[malloc_i].ptr != NULL) {
            malloc_i = free_i;
            while (malloc_i > 0 ) {
                malloc_i--;
                if (list->arr[malloc_i].func == FUNC_MALLOC &&
                    list->arr[malloc_i].ptr < list->arr[free_i].ptr &&
                    list->arr[malloc_i].ptr + list->arr[malloc_i].mem_size >
                    list->arr[free_i].ptr) {

                    list->arr[malloc_i].ptr = NULL;
                    list->arr[free_i].ptr = NULL;
                    break;
                } 
            }

        }

        // Did we not find the next free? Then look for it now.
        if (next_free_i == 0 && malloc_i > 0) {
            next_free_i = malloc_i;
            while (next_free_i > 0) {
                --next_free_i;
                if (list->arr[next_free_i].func == FUNC_FREE &&
                    list->arr[next_free_i].ptr != NULL) {
                    break;
                }
            }
        }
    }
    return true;
}

bool compress_merge(chimp_list_t *list)
{
    size_t i, j;

    for (i=0; i < list->size-1; ++i) {
        j = i;
        do {
            j++;
            if (list->arr[i].ptr + list->arr[i].mem_size == list->arr[j].ptr) {

                list->arr[i].mem_size += list->arr[j].mem_size;
                list->arr[j].ptr = NULL;
            }
        } while (list->arr[j].ptr == NULL);
        i = j;
    }
    return true;
}

/* remove uninteresting entries in the list
 *
 **/
bool chimp_list_compress(chimp_list_t *list)
{
    size_t get,put;
    size_t rm_cnt = 0;


    if (list->size == 0)
        return true;

    if (!compress_freed(list)) {
        return false;
    }
    
    //if (!compress_merge(list)) {
    //    return false;
    //}

    // Now we can remove all previously found malloc/free pairs from the array
    put = 0;
    for (get = 0; get < list->size; ++get) {
        if (list->arr[get].ptr != NULL) {
            list->arr[put++] = list->arr[get];
        }
    }
    list->size = put;
    return true;
}


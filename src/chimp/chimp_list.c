
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
    return true;
}

void chimp_list_free(chimp_list_t *list)
{
    list->ops->free(list->arr);
    list->alloc_size = 0;
    list->size = 0;
}

bool chimp_list_add_elem(chimp_list_t *list, chimp_list_elem_t elem)
{
    if (list->alloc_size <= list->size) {
        if (!(list->arr = list->ops->realloc(list->arr, list->alloc_size + 
                      CHIMP_LIST_INIT_SIZE*sizeof(chimp_list_elem_t))) ) {
            fprintf(stderr, "chimp_list: realloc failed!\n");
            return false;
        }
        list->alloc_size += CHIMP_LIST_INIT_SIZE;
    }
    list->arr[list->size++] = elem;
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
bool chimp_list_compress(chimp_list_t *list);

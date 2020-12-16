#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "log.h"
#include "list.h"
#include "api-recorder.h"

#define INITIAL_CAPACITY 4

int list_init(list *l, size_t element_size)
{
    if (l == NULL) {
        LOGE(LOG_ERROR, "list parameter is NULL");
        return 1;
    }
    if (element_size == 0LL) {
        LOGE(LOG_ERROR, "element_size of 0 does not make sense");
        return 1;
    }
    memset(l, 0, sizeof(list));
    if ((l->elements = malloc(INITIAL_CAPACITY*element_size)) == NULL) {
        LOGE(LOG_ERROR, "allocation failed");
        return 1;
    }
    l->element_size = element_size;
    l->capacity = INITIAL_CAPACITY;
    l->length = 0LL;

    return 0;
}

int list_free(list *l)
{
    if (l == NULL) {
        LOGE(LOG_ERROR, "list parameter is NULL");
        return 1;
    }
    free(l->elements);
    l->length = 0;
    l->capacity = 0;
    return 0;
}

int list_free_elements(list *l)
{
    if (l == NULL) {
        LOGE(LOG_ERROR, "list parameter is NULL");
        return 1;
    }
    for (size_t i=0; i < l->length; ++i) {
        free(*(void**)list_get(l, i));
    }
    return 0;
}

int list_append(list *l, void **new_element)
{
    int ret = 0;
    if (l == NULL) {
        LOGE(LOG_ERROR, "list parameter is NULL");
        return 1;
    }
    if (l->capacity == l->length) {
        l->elements = realloc(l->elements, l->capacity*2*l->element_size);
        if (l->elements == NULL) {
            LOGE(LOG_ERROR, "realloc failed.");
            /* the old pointer remains valid */
            return 1;
        }
        l->capacity *= 2;
    }
    if (new_element != NULL) {
        *new_element = list_get(l, l->length++);
    }

    return ret;
}

int list_append_copy(list *l, void *new_element)
{
    int ret = 0;
    void *elem;
    if ( (ret = list_append(l, &elem)) != 0) {
        goto out;
    }
    memcpy(elem, new_element, l->element_size);
 out:
    return ret;
}

int list_at(list *l, size_t at, void **element)
{
    if (l == NULL) {
        LOGE(LOG_ERROR, "list parameter is NULL");
        return 1;
    }
    if (at >= l->length) {
        LOGE(LOG_ERROR, "accessing list out of bounds");
        return 1;
    }
    if (element != NULL) {
        *element = list_get(l, at);
    }
    return 0;
}

inline void* list_get(list *l, size_t at) {
    return (l->elements+at*l->element_size);
}

int list_insert(list *l, size_t at, void *new_element)
{
    if (l == NULL) {
        LOGE(LOG_ERROR, "list parameter is NULL");
        return 1;
    }
    if (at > l->length) {
        LOGE(LOG_ERROR, "accessing list out of bounds");
        return 1;
    }
    if (at == l->length) {
        return list_append_copy(l, new_element);
    }

    if (list_append(l, NULL) != 0) {
        LOGE(LOG_ERROR, "error while lengthening list");
        return 1;
    }
    memmove(list_get(l, at+1), list_get(l, at), (l->length-at)*l->element_size);

    if (new_element != NULL) {
        memcpy(list_get(l, at), new_element, l->element_size);
    }

    l->length += 1; //appending a NULL element does not increase list length
    return 0;
}

int list_rm(list *l, size_t at)
{
    if (l == NULL) {
        LOGE(LOG_ERROR, "list parameter is NULL");
        return 1;
    }
    if (at >= l->length) {
        LOGE(LOG_ERROR, "accessing list out of bounds");
        return 1;
    }
    if (at < l->length-1) {
        memmove(list_get(l, at), list_get(l, at+1), (l->length-1-at)*l->element_size);
    }
    l->length -= 1;
    return 0;
}

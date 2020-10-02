#include <stdlib.h>
#include <string.h>

#include "log.h"
#include "list.h"

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
    for (size_t i=0; i < l->length; ++i) {
        free(l->elements[i]);
    }
    free(l->elements);
    l->length = 0;
    l->capacity = 0;
    return 0;
}

int list_append(list *l, void *new_element)
{
    if (l == NULL) {
        LOGE(LOG_ERROR, "list parameter is NULL");
        return 1;
    }
    if (l->capacity == l->length) {
        l->elements = realloc(l->elements, l->capacity*2);
        if (l->elements == NULL) {
            LOGE(LOG_ERROR, "realloc failed.");
            /* the old pointer remains valid */
            return 1;
        }
        l->capacity *= 2;
    }
    l->elements[l->length++] = new_element;

    return 0;
}

int list_alloc_append(list *l, void **new_element)
{
    if (l == NULL) {
        LOGE(LOG_ERROR, "list parameter is NULL");
        return 1;
    }
    if (new_element == NULL) {
        LOGE(LOG_ERROR, "new_element is NULL");
        return 1;
    }
    *new_element = malloc(l->element_size);
    if (*new_element == NULL) {
        LOGE(LOG_ERROR, "allocation failed.");
        return 1;
    }
    return list_append(l, *new_element);
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
        *element = l->elements[at];
    }
    return 0;
}

int list_rm(list *l, size_t at, void **element)
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
        *element = l->elements[at];
    }
    for (size_t i=at; i < l->length-1; i++) {
        l->elements[i] = l->elements[i+1];
    }
    l->length -= 1;
    return 0;
}

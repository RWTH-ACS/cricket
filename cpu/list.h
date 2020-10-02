#ifndef _LIST_H_
#define _LIST_H_
#include <unistd.h>

typedef struct list_t {
    void **elements;
    size_t length;
    size_t capacity;
    size_t element_size;
} list;

int list_init(list *l, size_t element_size);
int list_free(list *l);

/** appends an element to the list.
 * the element is not copied. It must be heap allocated.
 * @l the list
 * @new_element the new list element
 * @return 0 on success; 1 if l is NULL or allocation failed
 */
int list_append(list *l, void *new_element);

/** appends a newly allocated element to the list.
 * the element is only allocated but not initialized in any way.
 * @l the list
 * @new_element points to the newly created element
 * @return 0 on success; 1 if l is NULL or allocation failed
 */
int list_alloc_append(list *l, void **new_element);

/** return list element at the specified index.
 * @l the list
 * @at the index of the element
 * @element a pointer which should point to the element
 * @return 0 on success; 1 if l is NULL or at is larger then the
 *         list length
 */
int list_at(list *l, size_t at, void **element);

/** removes an element from the list.
 * if element is NULL, the element is freed.
 * @l the list
 * @at the index of the element to be removed
 * @element the element that was removed in returned in this pointer.
 * @return 0 on success; 1 if l is NULL or at is larger then the list
 *         length
 */
int list_rm(list *l, size_t at, void **element);

#endif //_LIST_H_

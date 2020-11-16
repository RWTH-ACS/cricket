#ifndef _LIST_H_
#define _LIST_H_
#include <unistd.h>

typedef struct list_t {
    void *elements;
    size_t length;
    size_t capacity;
    size_t element_size;
} list;

int list_init(list *l, size_t element_size);
int list_free(list *l);
int list_free_elements(list *l);

/** appends an element to the list.
 * the list is extended if needed.
 * @l the list
 * @new_element the new list element
 * @return 0 on success; 1 if l is NULL or allocation failed
 */
int list_append(list *l, void **new_element);

/** appends an element and copies its content.
 * the new element is copied using element_size as set during initialization.
 * @l the list
 * @new_element element from which to copy
 * @return 0 on success; 1 if l is NULL or allocation failed
 */
int list_append_copy(list *l, void *new_element);

/** return list element at the specified index.
 * @l the list
 * @at the index of the element
 * @element a pointer which should point to the element
 * @return 0 on success; 1 if l is NULL or at is larger then the
 *         list length
 */
int list_at(list *l, size_t at, void **element);

/** returns the position at which the requested element should be.
 * performs no checks - use with caution
 * @l the list
 * @at the index of the element
 * @return the element address or NULL on error
 */
void *list_get(list *l, size_t at);

/** removes an element from the list.
 * if element is NULL, the element is freed.
 * @l the list
 * @at the index of the element to be removed
 * @element the element that was removed in returned in this pointer.
 * @return 0 on success; 1 if l is NULL or at is larger then the list
 *         length
 */
int list_rm(list *l, size_t at);

/** inserts an element at the given position and copies its content.
 * the new element is copied using element_size as set during initialization.
 * Elements with indexes greater than @at are moved by one index.
 * @l the list
 * @at the position of the new element
 * @new_element element from which to copy
 * @return 0 on success; 1 if l is NULL or allocation failed
 */
int list_insert(list *l, size_t at, void *new_element);

#endif //_LIST_H_

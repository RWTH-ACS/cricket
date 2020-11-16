#ifndef _RESOURCE_MG_H_
#define _RESOURCE_MG_H_

#include "list.h"

typedef struct resource_mg_map_elem_t {
    void* client_address;
    void* cuda_address;
} resource_mg_map_elem;

typedef struct resource_mg_t {
    /* Restored resources where client address != cuda address
     * are stored here. This is a sorted list, enabling binary searching.
     * It contains elements of type resource_mg_map_elem
     */
    list map_res;
    /* During this run created resources where we use actual addresses on
     * the client side. This is an unordered list. We never have to search
     * this though. It containts elements of type void*.
     */
    list new_res;
    int bypass;
} resource_mg;

/** initializes the resource manager
 *
 * @bypass: if unequal to zero, searches for resources
 * will be bypassed, reducing the overhead. This is useful
 * for the original launch of an application as resources still
 * use their original pointers
 * @return 0 on success
 **/
int resource_mg_init(resource_mg *mg, int bypass);
void resource_mg_free(resource_mg *mg);

int resource_mg_add_sorted(resource_mg *mg, void* client_address, void* cuda_address);
int resource_mg_create(resource_mg *mg, void* cuda_address);

void* resource_mg_get(resource_mg *mg, void* client_address);

#endif //_RESOURCE_MG_H_

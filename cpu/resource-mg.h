#ifndef _RESOURCE_MG_H_
#define _RESOURCE_MG_H_

#include "list.h"

typedef struct resource_mg_map_elem_t {
    void* client_address;
    void* cuda_address;
} resource_mg_map_elem;

typedef struct memory_mg_map_elem_t {
    void* client_address;
    void* cuda_address;
    size_t size;
} memory_mg_map_elem;

typedef struct memory_mg_new_elem_t {
    void* cuda_address;
    size_t size;
} memory_mg_new_elem;

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

typedef struct memory_mg_t {
    resource_mg rm;
} memory_mg;


//Runtime API RMs
resource_mg rm_streams;
resource_mg rm_events;
resource_mg rm_arrays;
memory_mg   rm_memory;
resource_mg rm_kernels;
resource_mg rm_graphs;

//Driver API RMs
resource_mg rm_modules;
resource_mg rm_functions;
resource_mg rm_globals;
resource_mg rm_devices;
resource_mg rm_contexts;

//Other RMs
resource_mg rm_cusolver;
resource_mg rm_cublas;
resource_mg rm_cublaslt;


//CUDNN RMs
resource_mg rm_cudnn;
resource_mg rm_cudnn_tensors;
resource_mg rm_cudnn_filters;
resource_mg rm_cudnn_tensortransform;
resource_mg rm_cudnn_poolings;
resource_mg rm_cudnn_activations;
resource_mg rm_cudnn_lrns;
resource_mg rm_cudnn_convs;
resource_mg rm_cudnn_backendds;


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

void resource_mg_print(resource_mg *mg);

int memory_mg_init(memory_mg *mg, int bypass);
void memory_mg_free(memory_mg *mg);
int memory_mg_add_sorted(memory_mg *mg, void* client_address, void* cuda_address, size_t size);
int memory_mg_create(memory_mg *mg, void* cuda_address, size_t size);
void* memory_mg_get(memory_mg *mg, void* client_address);
void memory_mg_print(memory_mg *mg);


#endif //_RESOURCE_MG_H_

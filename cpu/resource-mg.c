#include "resource-mg.h"
#include "list.h"
#include "log.h"


int resource_mg_init(resource_mg *mg, int bypass)
{
    int ret = 0;
    if ((ret = list_init(&mg->new_res, sizeof(void*))) != 0) {
        goto out;
    }
    if (bypass == 0) {
        if ((ret = list_init(&mg->map_res, sizeof(resource_mg_map_elem))) != 0) {
            goto out;
        }
    }
    mg->bypass = bypass;
 out:
    return ret;
}

void resource_mg_free(resource_mg *mg)
{
    list_free(&mg->new_res);
    if (mg->bypass == 0) {
        list_free(&mg->map_res);
    }
}

int resource_mg_create(resource_mg *mg, void *cuda_address)
{
    if (list_append_copy(&mg->new_res, cuda_address) != 0) {
        LOGE(LOG_ERROR, "failed to append to new_res");
        return 1;
    }
    return 0;
}

static void* resource_mg_search_map(resource_mg *mg, void *client_address)
{
    size_t start = 0;
    size_t end = mg->map_res.length;
    size_t mid;
    
    while (end >= start) {
        mid = start + (end-start)/2;
        resource_mg_map_elem *mid_elem = list_get(&mg->map_res, mid);
        if (mid_elem == NULL) {
            LOG(LOG_ERROR, "list state of map_res is inconsistent\n");
            return NULL;
        }

        if (mid_elem->client_address > client_address) {
            end = mid-1;
        } else if (mid_elem->client_address < client_address) {
            start = mid+1;
        } else /*if (mid_elem->client_address == client_address)*/ {
            return mid_elem->cuda_address;
        }
    }
    return NULL;
}

inline void* resource_mg_get(resource_mg *mg, void* client_address)
{
    if (mg->bypass) {
        return client_address;
    } else {
        return resource_mg_search_map(mg, client_address);
    }
    return 0;
}

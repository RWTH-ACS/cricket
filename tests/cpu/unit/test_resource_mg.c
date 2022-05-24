#include <stdio.h>
#include <assert.h>
#include "resource-mg.h"
#include "log.h"


int main()
{
    resource_mg rm1;
    resource_mg rm2;
    int ret;
    void* address;

    ret = resource_mg_init(&rm1, 1);
    assert(ret == 0 && rm1.bypass == 1);
    ret = resource_mg_init(&rm2, 0);
    assert(ret == 0 && rm2.bypass == 0);

    for(void *i=0; i < (void*)10; ++i) {
        ret = resource_mg_create(&rm1, i);
        assert(ret == 0);
        address = resource_mg_get(&rm1, i);
        assert(address == i);
    }
    for(void *i=0; i < (void*)10; i+=2) {
        ret = resource_mg_create(&rm2, i);
        assert(ret == 0);
        address = resource_mg_get(&rm2, i);
        assert(address == i);
    }

    for(void *i=0; i < (void*)10; ++i) {
        address = resource_mg_get(&rm1, i);
        assert(address == i);
    }
    for(void *i=0; i < (void*)10; i+=2) {
        address = resource_mg_get(&rm2, i);
        assert(address == i);
    }

    ret = resource_mg_add_sorted(&rm1, (void*)1, (void*)1);
    assert(ret != 0);

    for(void *i=0; i < (void*)10; i+=2) {
        ret = resource_mg_add_sorted(&rm2, i, i+1000);
        assert(ret == 0);
        address = resource_mg_get(&rm2, i);
        assert(address == i+1000);
    }
    for(void *i=0; i < (void*)10; i+=2) {
        address = resource_mg_get(&rm2, i);
        assert(address == i+1000);
    }
    for(void *i=(void*)1; i < (void*)10; i+=2) {
        address = resource_mg_get(&rm2, i);
        assert(address == (void*)i);
    }

    resource_mg_free(&rm1);
    resource_mg_free(&rm2);
    LOG(LOG_INFO, "resource_mg passed.");
    return 0;
}

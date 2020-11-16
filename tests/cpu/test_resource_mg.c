#include <stdio.h>
#include <assert.h>
#include "resource-mg.h"
#include "log.h"


//int resource_mg_create(resource_mg *mg, void* cuda_address);
//int resource_mg_restore(resource_mg *mg, void* cuda_address, void* client_address);

void* resource_mg_get(resource_mg *mg, void* client_address);
int main()
{
    resource_mg rm1;
    resource_mg rm2;
    int ret;

    ret = resource_mg_init(&rm1, 1);
    assert (ret == 0);
    ret = resource_mg_init(&rm2, 0);
    assert (ret == 0);




    resource_mg_free(&rm1);
    resource_mg_free(&rm2);
    LOG(LOG_INFO, "resource_mg passed.");
    return 0;
}

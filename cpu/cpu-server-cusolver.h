#ifndef _CPU_SERVER_CUSOLVER_H_
#define _CPU_SERVER_CUSOLVER_H_

#include "resource-mg.h"

int cusolver_init(int restore, resource_mg *streams, resource_mg *memory);
resource_mg *cusolver_get_rm(void);
int cusolver_deinit(void);

#endif

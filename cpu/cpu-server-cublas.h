#ifndef _CPU_SERVER_CUBLAS_H_
#define _CPU_SERVER_CUBLAS_H_

#include "resource-mg.h"

int cublas_init(int restore, resource_mg *memory);
resource_mg *cublas_get_rm(void);
int cublas_deinit(void);

#endif

#ifndef _CPU_SERVER_CUBLASLT_H_
#define _CPU_SERVER_CUBLASLT_H_

#include "resource-mg.h"

int cublaslt_init(int restore, resource_mg *memory);
int cublaslt_deinit(void);
resource_mg *cublaslt_get_rm(void);

#endif // _CPU_SERVER_CUBLASLT_H_

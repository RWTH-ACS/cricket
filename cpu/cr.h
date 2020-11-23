#ifndef _CR_H_
#define _CR_H_
#include "cpu_rpc_prot.h"
#include "resource-mg.h"

int cr_dump(const char *path);
int cr_restore(const char *path, resource_mg *rm_memory, resource_mg *rm_streams);

#endif //_CR_H_

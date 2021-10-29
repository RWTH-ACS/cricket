#ifndef _CR_H_
#define _CR_H_
#include "cpu_rpc_prot.h"
#include "resource-mg.h"

int cr_dump(const char *path);
int cr_dump_memory(const char *path);
int cr_dump_rpc_id(const char *path, unsigned long prog, unsigned long vers);
int cr_restore_rpc_id(const char *path, unsigned long *prog, unsigned long *vers);
int cr_restore(const char *path, resource_mg *rm_memory, resource_mg *rm_streams, resource_mg *rm_events, resource_mg *rm_arrays, resource_mg *rm_cusolver, resource_mg *rm_cublas);

#endif //_CR_H_

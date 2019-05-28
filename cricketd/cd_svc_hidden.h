#ifndef _CD_SVC_HIDDEN_H_
#define _CD_SVC_HIDDEN_H_


/* this file is for the hidden function exported from some driver
 * library (presumably) via the cuGetExportTable API (which is undocumented).
 * During initialization cuGetExportTable is called 4 times with different
 * parameters. Each call seems to copy a varying amount of function pointers
 * to the first paramter. The memory for these function pointers is
 * callee allocated and the first element is the length in bytes of the 
 * entire list of pointers. Additionally the last element is NULL.
 * The last call does not provide a length element (Nvidia seems
 * unessecary cruel here)
 *
 * The first call to cuGetExportTable exports 8 function pointers.
 *
 * The second call exports 6 function pointers.
 *
 * The third call exports 2 function pointers.
 *
 * The fourth call exports 3 function pointers.
 *
 * I guess we can assume even these hidden functions adhere to nvidias
 * code style of functions always return cuResult, with 0 being success.
 *
 * To circument thes ugly and mean hack we do a bit of hacking ourselved:
 */

void* cd_svc_hidden_add_table(void* export_table, size_t len);
void *cd_svc_hidden_get(size_t call, size_t index);


#endif //_CD_SVC_HIDDEN_H_

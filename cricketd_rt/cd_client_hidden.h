#ifndef _CD_CLIENT_HIDDEN_H_
#define _CD_CLIENT_HIDDEN_H_


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
 * To circument thes ugly and mean hack we do a bit of hacking ourselves:
 */

//void* cd_client_hidden_replace(void* orig_addr, size_t index);
void cd_client_hidden_init(void *new_clnt);
void cd_client_hidden_reset(void);
int cd_client_hidden_incr(void);
void *cd_client_hidden_get(void *orig_ptr);
void *cd_client_hidden_orgi_ptr(void *replaced_ptr);

void* cd_client_get_real_ctx(void* fake_ctx);
void* cd_client_get_fake_ctx(void* real_ctx);
void* cd_client_get_real_module(void* fake_module);
void* cd_client_get_fake_module(void* real_module);

int hidden_0_0(void* arg1);
int hidden_get_device_ctx(void** cu_ctx, int cu_device);
int hidden_0_2(void* arg1);
int hidden_0_3(void* arg1);
int hidden_0_4(void* arg1);
int hidden_get_module(void** arg1, void** arg2, void* arg3, void* arg4, int arg5);
int hidden_0_6(void* arg1);
int hidden_0_7(void* arg1);

int hidden_1_0(int arg1, void* arg2);
int hidden_1_1(void* arg1, void *arg2);
int hidden_1_2(void* arg1);
int hidden_1_3(void* arg1, void* arg2);
int hidden_1_4(void* arg1);
int hidden_1_5(void* arg1, void* arg2);

int hidden_2_0(void* arg1);
int hidden_2_1(void* arg1);

int hidden_3_0(int arg1, void** arg2, void** arg3);
int hidden_3_1(void* arg1, void* arg2);
int hidden_3_2(void** arg1, int arg2, void** arg3);

#endif //_CD_CLIENT_HIDDEN_H_

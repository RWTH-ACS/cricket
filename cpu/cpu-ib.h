#ifndef _CPU_IB_H_
#define _CPU_IB_H_


int ib_init(int _device_id);
int ib_connect_server(void *memreg, int mr_id);
int ib_connect_client(void *memreg, int mr_id, char *server_address);
void ib_free_memreg(void* memreg, int mr_id);
void ib_cleanup(void);
void ib_final_cleanup(void);
size_t ib_allocate_memreg(void** mem_address, size_t memsize, int mr_id);
int ib_server_recv(void *memptr, int mr_id, size_t length);
int ib_client_send(void *memptr, int mr_id, size_t length, char *peer_node);

#endif //_CPU_IB_H_

#ifndef _CPU_SERVER_H_
#define _CPU_SERVER_H_

#include <stddef.h>

void cricket_main(char* app_command, size_t prog_version, size_t vers_num);
void cricket_main_hash(char* app_command);
void cricket_main_static(size_t prog_num, size_t vers_num);

#endif //_CPU_SERVER_H_

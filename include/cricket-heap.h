#ifndef _CRICKET_HEAP_H_
#define _CRICKET_HEAP_H_

#include "cudadebugger.h"
#include <stddef.h>

// set gdb focus to host to read host's variables and stuff
bool cricket_focus_host(bool batch_flag);

// set gdb focus to kernel to read GPU variables and stuff
bool cricket_focus_kernel(bool batch_flag);

// Checks the size of the allocated memory at the address addr.
// Returns false if the memory is not allocated at all.
// Stores the size in *size
bool cricket_heap_memreg_size(void *addr, size_t *size);

#endif //_CRICKET_HEAP_H_

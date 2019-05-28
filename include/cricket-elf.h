#ifndef _CRICKET_ELF_H_
#define _CRICKET_ELF_H_

#include "cricket-types.h"
#include "cudadebugger.h"
#include <stddef.h>

#define CRICKET_SASS_NOP 0x50b0000000070f00
#define CRICKET_SASS_JMX(REG) (0xe20000000007000f | ((REG & 0xff) << 8))
#define CRICKET_SASS_BRX(REG) (0xe25000000007000f | ((REG & 0xff) << 8))
#define CRICKET_SASS_BRX_PRED(REG) (0xe25000000000000f | ((REG & 0xff) << 8))
#define CRICKET_SASS_SSY(ADDR)                                                 \
    (0xe290000000000000 | ((ADDR & 0xffffffff) << 20))
#define CRICKET_SASS_PRET(ADDR)                                                \
    (0xe270000000000040 | ((ADDR & 0xffffffff) << 20))
#define CRICKET_SASS_SYNC(PRED) (0xf0f800000000000f | ((PRED & 0xf) << 20))
#define CRICKET_SASS_EXIT 0xe30000000007000f
#define CRICKET_SASS_CONTROL ((0x5L << 42) + (0x5L << 21) + 0x5L)
#define CRICKET_SASS_FCONTROL 0x001ffc00fd4007ef

void cricket_elf_free_info(cricket_elf_info *info);

bool cricket_elf_get_info(const char *function_name, cricket_elf_info *info);

bool cricket_elf_get_sass_info(const char *filename, const char *section_name,
                               uint64_t relative_pc, cricket_sass_info *info);

bool cricket_elf_restore_patch(const char *filename, const char *new_filename,
                               cricket_callstack *callstack);
bool cricket_elf_get_global_vars_info(cricket_global_var **globals,
                                      size_t *globals_size);

bool cricket_elf_pc_info(const char *function_name, uint64_t relative_pc,
                         uint64_t *relative_ssy, uint64_t *relative_pbk);

bool cricket_elf_analyze(const char *filename);

bool cricket_elf_patch_all(const char *filename, const char *new_filename,
                           cricket_jmptable_index **jumptable,
                           size_t *jumptable_len);

bool
cricket_elf_build_fun_info(cricket_function_info_array *function_info_array);

cricket_function_info *
cricket_elf_get_fun_info(const cricket_function_info_array *fi_array,
                         const char *fun_name);

void cricket_elf_free_jumptable(cricket_jmptable_index **jmptbl,
                                size_t jmptbl_len);

bool cricket_elf_get_jmptable_index(cricket_jmptable_index *jmptbl,
                                    size_t jmptbl_len, const char *fn,
                                    cricket_jmptable_index **entry);
bool cricket_elf_get_jmptable_addr(cricket_jmptable_entry *entries,
                                   size_t entries_num, uint64_t destination,
                                   uint64_t *address);

#endif //_CRICKET_ELF_H_

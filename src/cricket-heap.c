#include "defs.h"
#include "command.h"
#include "cli/cli-cmds.h"
#include "value.h"
#include <assert.h>

#include "cricket-heap.h"
#include "cricket-utils.h"

bool cricket_focus_host(bool batch_flag)
{
    struct cmd_list_element *c;
    char *threadcmd = "thread 1";
    c = lookup_cmd(&threadcmd, cmdlist, "", 0, 1);

    if (c == NULL || c == (struct cmd_list_element *)-1)
        return false;

    if (!cmd_func_p(c))
        return false;

    cmd_func(c, threadcmd, !batch_flag);
    return true;
}

bool cricket_focus_kernel(bool batch_flag)
{
    struct cmd_list_element *c;
    char *threadcmd = "cuda kernel 0";
    c = lookup_cmd(&threadcmd, cmdlist, "", 0, 1);

    if (c == NULL || c == (struct cmd_list_element *)-1)
        return false;

    if (!cmd_func_p(c))
        return false;

    cmd_func(c, threadcmd, !batch_flag);
    return true;
}

bool cricket_heap_memreg_size(void *addr, size_t *size)
{
    char *callstr = NULL;
    struct value *val;
    struct type *type;

    assert(size != NULL);

    if (addr == NULL)
        return false;

    // Call the function getSize in the CUDA kernel
    if (asprintf(&callstr, "getSize(%p)", addr) == -1)
        return false;
    *size = parse_and_eval_long(callstr);
    free(callstr);

    // No memory is allocated at the given address
    if (*size == 0)
        return false;

    return true;
}

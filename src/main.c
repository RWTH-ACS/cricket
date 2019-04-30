#include "defs.h"
#include <stdio.h>
#include "cudadebugger.h"
#include "cuda-api.h"
#include "cuda-options.h"
#include "inferior.h"
#include "top.h"
#include "bfd.h"
#include "cuda-tdep.h"
#include "objfiles.h"
#include "cuda-state.h"
#include "gdbcore.h"
#include "regcache.h"
#include "cli/cli-cmds.h"
#include "cli/cli-setshow.h"
#include "interps.h"
#include "main.h"
#include <dlfcn.h>
#include <sys/time.h>

/* #include "cricket-cr.h" */
/* #include "cricket-device.h" */
/* #include "cricket-stack.h" */
/* #include "cricket-register.h" */
/* #include "cricket-file.h" */
/* #include "cricket-heap.h" */
#include "cricket-elf.h"
/* #include "cricket-utils.h" */
#include "cricket-checkpoint.h"
#include "cricket-restore.h"

#define CRICKET_PROFILE 1

CUDBGAPI cudbgAPI = NULL;

// in defs.h these variables are refferd to as external, so let's provide these
// as global state. (What should possibly go wrong)
struct ui_file *gdb_stdout;
struct ui_file *gdb_stderr;
struct ui_file *gdb_stdlog;
struct ui_file *gdb_stdin;
struct ui_file *gdb_stdtarg;
struct ui_file *gdb_stdtargerr;
struct ui_file *gdb_stdtargin;

bool cricket_init_gdb(char *name)
{
    // TODO check if tools succeed
    /* initialize gdb streams, necessary for gdb_init */
    gdb_stdout = ui_file_new();
    gdb_stderr = stdio_fileopen(stderr);
    gdb_stdlog = gdb_stderr;
    gdb_stdtarg = gdb_stderr;
    gdb_stdin = stdio_fileopen(stdin);
    gdb_stdtargerr = gdb_stderr;
    gdb_stdtargin = gdb_stdin;
    instream = fopen("/dev/null", "r");

    /* initialize gdb paths */
    gdb_sysroot = strdup("");
    debug_file_directory = strdup(DEBUGDIR);
    gdb_datadir = strdup(GDB_DATADIR);

    /* tell gdb that we do not want to run an interactive shell */
    batch_flag = 1;

    /* initialize BFD, the binary file descriptor library */
    bfd_init();

    /* initialize GDB */
    printf("gdb_init...\n");
    gdb_init(name);

    char *interpreter_p = strdup(INTERP_CONSOLE);
    struct interp *interp = interp_lookup(interpreter_p);
    interp_set(interp, 1);
    return true;
}

// TODO: remove, this is basically a wrapper for elf_analyze
int cricket_analyze(int argc, char *argv[])
{
    if (argc != 3) {
        printf("wrong number of arguments, use: %s <executable>\n", argv[0]);
        return -1;
    }
    printf("Analyzing \"%s\"\n", argv[2]);
    if (!cricket_elf_analyze(argv[2])) {
        printf("cricket analyze unsuccessful\n");
        return -1;
    }
    return 0;
}

int cricket_start(int argc, char *argv[])
{
    CUDBGResult res;
    if (argc != 3) {
        printf("wrong number of arguments, use: %s <executable>\n", argv[0]);
        return -1;
    }

    cricket_init_gdb(argv[0]);

    /* load files */
    exec_file_attach(argv[2], !batch_flag);
    symbol_file_add_main(argv[2], !batch_flag);

    struct cmd_list_element *c;
    char *pset = "set cuda break_on_launch all";
    c = lookup_cmd(&pset, cmdlist, "", 0, 1);
    do_set_command(pset, !batch_flag, c);

    char *prun = "run";
    c = lookup_cmd(&prun, cmdlist, "", 0, 1);
    cmd_func(c, prun, !batch_flag);

detach:
    /* Detach from process (CPU and GPU) */
    detach_command(NULL, !batch_flag);
    /* quit GDB. TODO: Why is this necccessary? */
    quit_force(NULL, 0);
    return 0;
}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        fprintf(stderr, "wrong number of arguments, use: %s "
                        "(start|checkpoint|restore)\n",
                argv[0]);
        return -1;
    }
    if (strcmp(argv[1], "start") == 0)
        return cricket_start(argc, argv);
    if (strcmp(argv[1], "checkpoint") == 0)
        return cricket_checkpoint(argc, argv);
    if (strcmp(argv[1], "restore") == 0 || strcmp(argv[1], "restart") == 0)
        return cricket_restore(argc, argv);
    if (strcmp(argv[1], "analyze") == 0)
        return cricket_analyze(argc, argv);

    fprintf(stderr, "Unknown operation \"%s\".\n", argv[1]);
    return -1;
}

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
#include <argp.h>

#include "cricket-elf.h"
#include "cricket-checkpoint.h"
#include "cricket-restore.h"
#include "cricket-utils.h"

// TODO make argument option
#define CRICKET_PROFILE 1

CUDBGAPI cudbgAPI = NULL;

struct arguments
{
    enum {
        START,
        CHECKPOINT,
        RESTORE,
        ANALYZE,
        ERROR
    } mode;
    char *executable;
    char *pid;
    char *ckp_dir;
    int profile;
};

const char *argp_program_version = "cricket 0.1.0";
const char *argp_program_bug_address = "https://git.rwth-aachen.de/"
                                       "niklas.eiling/cricket";
static char doc[] = "cricket - Checkpoint-Restart in Cuda KErnels Tool";
static char args_doc[] = "";

static struct argp_option options
    [] = { { "\bProvide exactly one of these:", 0, 0,
             OPTION_DOC | OPTION_NO_USAGE,   0, 0 },
           { "analyze", 'a', "executable", 0, "Analyze something TODO!", 1 },
           { "restore", 'r', "executable", 0, "Restore a kernel from a "
                                              "checkpoint. Also consider using "
                                              "-d option",
             1 },
           { "restart", 'r', "executable", OPTION_ALIAS },
           { "start", 's', "executable", 0, "Start a CUDA application", 1 },
           { "run", 0, "executable", OPTION_ALIAS },
           { "checkpoint", 'c', "pid", 0, "Checkpoint a running CUDA "
                                          "application. Also consider "
                                          "using -d option",
             1 },
           { "\bOther:", 0, 0, OPTION_DOC | OPTION_NO_USAGE, 0, 2 },
           { "dir", 'd', "checkpoint-directory", 0,
             "specifies the directory for the "
             "checkpoint file. If not given, "
             "/tmp/cricket will be used",
             3 },
           { "profile",                                                   'p',
             0,                                                           0,
             "display the time spend in different stages of the program", 3 },
           { 0 } };

static error_t parse_opt(int key, char *arg, struct argp_state *state)
{
    struct arguments *arguments = state->input;
    switch (key) {
    case ARGP_KEY_INIT:
        arguments->mode = ERROR;
        arguments->executable = "";
        arguments->pid = "-1";
        arguments->ckp_dir = "/tmp/cricket";
        arguments->profile = 0;
        break;
    case 'a':
        arguments->mode = ANALYZE;
        arguments->executable = arg;
        break;
    case 'r':
        arguments->mode = RESTORE;
        arguments->executable = arg;
        break;
    case 's':
        arguments->mode = START;
        arguments->executable = arg;
        break;
    case 'c':
        arguments->mode = CHECKPOINT;
        arguments->pid = arg;
        break;
    case 'd':
        arguments->ckp_dir = arg;
        break;
    case 'p':
        arguments->profile = 1;
        break;
    case ARGP_KEY_ARG:
        // No further arguments allowed
        return ARGP_ERR_UNKNOWN;
        break;
    case ARGP_KEY_END:
        if (arguments->mode == ERROR) {
            argp_usage(state);
        }
        break;
    default:
        return ARGP_ERR_UNKNOWN;
    }
    return 0;
}

static struct argp argp = { options, parse_opt, args_doc, doc, 0, 0, 0 };


int cricket_start(char *executable)
{
    CUDBGResult res;

    /* load files */
    exec_file_attach(executable, !batch_flag);
    symbol_file_add_main(executable, !batch_flag);

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
    struct arguments cricket_args;
    argp_parse(&argp, argc, argv, 0, 0, &cricket_args);
    if (cricket_args.mode == ERROR) {
        fprintf(stderr, "Argument parsing error\n");
        return -1;
    }

    switch (cricket_args.mode) {
    case ANALYZE:
        printf("Analyzing \"%s\"\n", cricket_args.executable);
        if (!cricket_elf_analyze(cricket_args.executable)) {
            fprintf(stderr, "cricket analyze unsuccessful\n");
            return -1;
        }
        return 0;
    case START:
        cricket_init_gdb(argv[0]);
        return cricket_start(cricket_args.executable);
    case CHECKPOINT:
        cricket_init_gdb(argv[0]);
        return cricket_checkpoint(cricket_args.pid, cricket_args.ckp_dir,
                                  cricket_args.profile);
    case RESTORE:
        return cricket_restore(cricket_args.executable, cricket_args.ckp_dir);
    default:
        fprintf(stderr, "unknown mode\n");
        return -1;
    }
}

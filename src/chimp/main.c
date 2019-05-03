#include <sys/ptrace.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/user.h>
#include <unistd.h>
#include <stdio.h>


int main (char **argv, int argc)
{
    pid_t pid = fork();
    if (pid == -1) {
        fprintf(stderr, "fork failed.\n");
    } else if (pid == 0) {
        ptrace(PTRACE_TRACEME, 0, 0, 0);
        execvp(argv[1], argv + 1);
        printf("have a nice day.\n");
        return 0;
    }

    waitpid(pid, 0, 0);
    ptrace(PTRACE_SETOPTIONS, pid, 0, 0);
    /* Enter next system call */
    ptrace(PTRACE_SYSCALL, pid, 0, 0);
    waitpid(pid, 0, 0);

    struct user_regs_struct regs;
    ptrace(PTRACE_GETREGS, pid, 0, &regs);
    long syscall = regs.orig_rax;

    fprintf(stderr, "%ld(%ld, %ld, %ld, %ld, %ld, %ld)",
            syscall,
            (long)regs.rdi, (long)regs.rsi, (long)regs.rdx,
            (long)regs.r10, (long)regs.r8,  (long)regs.r9);

    ptrace(PTRACE_SYSCALL, pid, 0, 0);
    waitpid(pid, 0, 0);

    ptrace(PTRACE_GETREGS, pid, 0, &regs);
    fprintf(stderr, " = %ld\n", (long)regs.rax);


    return 0;
}


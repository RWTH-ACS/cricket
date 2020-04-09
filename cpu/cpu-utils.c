#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <sys/wait.h>
#include <openssl/md5.h>

#include <bfd.h>

#include "cpu-utils.h"
#include "cpu-common.h"

#define CRICKET_ELF_NV_INFO_PREFIX ".nv.info"
#define CRICKET_ELF_NV_SHARED_PREFIX ".nv.shared."
#define CRICKET_ELF_NV_TEXT_PREFIX ".nv.text."
#define CRICKET_ELF_TEXT_PREFIX ".text."

#define CRICKET_ELF_FATBIN ".nv_fatbin"
#define CRICKET_ELF_REGFUN "_ZL24__sti____cudaRegisterAllv"

int cpu_utils_md5hash(char *filename, unsigned long *high, unsigned long *low)
{
    unsigned char c[MD5_DIGEST_LENGTH];
    FILE *fd;
    MD5_CTX mdContext;
    int bytes;
    unsigned char data[1024];

    if (filename == NULL || high == NULL || low == NULL) {
        return -1;
    }

    if ((fd = fopen(filename, "rb")) == NULL) {
        printf ("%s can't be opened.\n", filename);
        return -1;
    }

    MD5_Init (&mdContext);
    while ((bytes = fread(data, 1, 1024, fd)) != 0)
        MD5_Update(&mdContext, data, bytes);
    MD5_Final(c, &mdContext);
    fclose (fd);
    *high = *((unsigned long*)c);
    *low  = *((unsigned long*)(c+8));
    return 0;
}

void* cricketd_utils_symbol_address(char *symbol)
{
    bfd *hostbfd = NULL;
    asection *section;
    FILE *hostbfd_fd = NULL;
    void *ret = NULL;
    size_t symtab_size, symtab_length;
    asymbol **symtab = NULL;


    bfd_init();

    if ((hostbfd_fd = fopen("/proc/self/exe", "rb")) == NULL) {
        fprintf(stderr, "cricketd (%d): fopen failed\n", __LINE__);
        return NULL;
    }

    if ((hostbfd = bfd_openstreamr("/proc/self/exe", NULL, hostbfd_fd)) == NULL) {
        fprintf(stderr, "cricketd (%d): bfd_openr failed on %s\n", __LINE__,
                "/proc/self/exe");
        fclose(hostbfd_fd);
        goto cleanup;
    }

    if (!bfd_check_format(hostbfd, bfd_object)) {
        fprintf(stderr, "cricketd (%d): %s has wrong bfd format\n", __LINE__,
                "/proc/self/exe");
        goto cleanup;
    }

    if ((symtab_size = bfd_get_symtab_upper_bound(hostbfd)) == -1) {
        fprintf(stderr, "cricketd: bfd_get_symtab_upper_bound failed\n");
        return NULL;
    }

    //printf("symtab size: %lu\n", symtab_size);

    if ((symtab = (asymbol **)malloc(symtab_size)) == NULL) {
        fprintf(stderr, "cricketd: malloc symtab failed\n");
        return NULL;
    }

    if ((symtab_length = bfd_canonicalize_symtab(hostbfd, symtab)) == 0) {
        //printf("symtab empty...\n");
    } else {
        //printf("%lu symtab entries\n", symtab_length);
    }

    for (int i = 0; i < symtab_length; ++i) {
        if (strcmp(bfd_asymbol_name(symtab[i]), CRICKET_ELF_REGFUN) == 0) {
            ret = (void*)bfd_asymbol_value(symtab[i]);
            break;
        }
        //printf("%d: %s: %lx\n", i, bfd_asymbol_name(symtab[i]),
        //       bfd_asymbol_value(symtab[i]));
    }


cleanup:
    free(symtab);
    if (hostbfd != NULL)
        bfd_close(hostbfd);
    return ret;
}

int cricketd_utils_launch_child(const char *file, char **args)
{
    int filedes[2];
    FILE *fd = NULL;

    if (pipe(filedes) == -1) {
        fprintf(stderr, "error while creating pipe\n");
        return -1;
    }

    pid_t pid = fork();
    if (pid == -1) {
        fprintf(stderr, "error while forking\n");
        return -1;
    } else if (pid == 0) {
        while ((dup2(filedes[1], STDOUT_FILENO) == -1) && (errno == EINTR)) {}
        close(filedes[1]);
        close(filedes[0]);
        char *env[] = {NULL};
        execvpe(file, args, env);
        exit(1);
    }
    close(filedes[1]);
    return filedes[0];
}

kernel_info_t* cricketd_utils_search_info(kernel_info_t *infos, size_t kernelnum, char *kernelname)
{
    if (infos == NULL || kernelname == NULL) {
        //fprintf(stderr, "error: wrong parameters (%d)\n", __LINE__);
        return NULL;
    }

    for (int i=0; i < kernelnum; ++i) {
        if (strcmp(kernelname, infos[i].name) == 0) {
            return &infos[i];
        }
    }
    return NULL;
}

int cricketd_utils_parameter_size(kernel_info_t **infos, size_t *kernelnum)
{
    char knamebuf[128] = {0};
    char type[64] = {0};
    size_t value;
    int res;
    char *args[] = {"cricket", "info", NULL, NULL};
    char linktarget[1024] = {0};
    char valuebuf[1024] = {0};
    int output;
    FILE *fdesc;
    kernel_info_t *buf = NULL;
    int ret = -1;
    int child_exit = 0;

    if (infos == NULL || kernelnum == NULL) {
        fprintf(stderr, "error: wrong parameters at %d\n", __LINE__);
        goto out;
    }
    *kernelnum = 0;
    *infos = NULL;

    if (readlink("/proc/self/exe", linktarget, 1024) == 1024) {
        fprintf(stderr, "error: executable path too long\n");
        goto out;
    }
    args[2] = linktarget;

    if ( (output = cricketd_utils_launch_child(CRICKET_PATH, args)) == -1) {
        fprintf(stderr, "error while launching child\n");
        goto out;
    }
    if ( (fdesc = fdopen(output, "r")) == NULL) {
        fprintf(stderr, "error while opening stream\n");
        close(output);
        goto out;
    }
    while (1) {
        if ( (res = fscanf(fdesc, "%128s %64s %1024s\n", knamebuf, type, valuebuf)) != 3) {
            if (feof(fdesc)) {
                break;
            } else if (ferror(fdesc)) {
                fprintf(stderr, "error while reading from pipe\n");
                goto cleanup;
            }
        } else {
            if (knamebuf == NULL) {
                continue;
            }
            buf = cricketd_utils_search_info(*infos, *kernelnum, knamebuf);
            //printf("kname: %s, type: %s, val: %s\n", knamebuf, type, valuebuf);
            if (buf == NULL) {
                if ((*infos = realloc(*infos, (++(*kernelnum))*sizeof(kernel_info_t))) == NULL) {
                    fprintf(stderr, "error: malloc failed (%d)\n", __LINE__);
                    goto cleanup;
                }
                buf = &((*infos)[(*kernelnum)-1]);
                memset(buf, 0, sizeof(kernel_info_t));
                if ((buf->name = malloc(strlen(knamebuf))) == NULL) {
                    fprintf(stderr, "error: malloc failed (%d)\n", __LINE__);
                    goto cleanup;
                }
                strcpy(buf->name, knamebuf);
            }
            if (strcmp("param_size", type) == 0) {
                if (sscanf(valuebuf, "%zu", &value) != 1) {
                    fprintf(stderr, "error (%d)\n", __LINE__);
                    goto cleanup;
                }
                buf->param_size = value;
            } else if (strcmp("param_num", type) == 0) {
                if (sscanf(valuebuf, "%zu", &value) != 1) {
                    fprintf(stderr, "error (%d)\n", __LINE__);
                    goto cleanup;
                }
                buf->param_num = value;
            } else if (strcmp("param_offsets", type) == 0) {
                buf->param_offsets = malloc(sizeof(uint16_t)*buf->param_num);
                for (int i=0; i < buf->param_num; ++i) {
                    if (sscanf(valuebuf, "%4x,%s", &value, valuebuf) < 1) {
                        fprintf(stderr, "error (%d)\n", __LINE__);
                        goto cleanup;
                    }
                    buf->param_offsets[i] = value;
                }
            } else if (strcmp("param_sizes", type) == 0) {
                buf->param_sizes = malloc(sizeof(uint16_t)*buf->param_num);
                for (int i=0; i < buf->param_num; ++i) {
                    if (sscanf(valuebuf, "%4x,%s", &value, valuebuf) < 1) {
                        fprintf(stderr, "error (%d)\n", __LINE__);
                        goto cleanup;
                    }
                    buf->param_sizes[i] = value;
                }
            }
        }
    }
    ret = 0;
 cleanup:
    fclose(fdesc);
    close(output);
    wait(&child_exit);
    printf("child_exit:%d\n", child_exit);
 out:
    return (ret != 0 ? ret : child_exit);
}

void kernel_infos_free(kernel_info_t *infos, size_t kernelnum)
{
    for (int i=0; i < kernelnum; ++i) {
        free(infos[i].name);
        free(infos[i].param_offsets);
    }
}

#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <sys/wait.h>
#include <linux/limits.h>
#include "rpc/types.h"
#include <sys/stat.h>

#include "cpu-utils.h"
#include "cpu-common.h"
#include "log.h"


int cpu_utils_command(char **command)
{
    FILE* fd;
    char str[128] = {0};
    size_t comm_len = 0;
    int ret = -1;
    if ((fd = fopen("/proc/self/comm", "r")) == NULL) {
        LOGE(LOG_ERROR, "%s can't be opened.", "/proc/self/comm");
        return -1;
    }

    if ((comm_len = fread(str, 1, 128, fd)) == 0) {
        LOGE(LOG_ERROR, "could not read from /proc/self/comm");
        goto cleanup;
    }

    if ((*command = realloc(*command, comm_len)) == NULL) {
        LOGE(LOG_ERROR, "realloc failed");
        goto cleanup;
    }
    
    strncpy(*command, str, comm_len-1);
    (*command)[comm_len-1] = '\0';
    ret = 0;
 cleanup:
    fclose(fd);
    return ret;

}


int cpu_utils_launch_child(const char *file, char **args)
{
    int filedes[2];
    FILE *fd = NULL;

    if (pipe(filedes) == -1) {
        LOGE(LOG_ERROR, "error while creating pipe");
        return -1;
    }

    pid_t pid = fork();
    if (pid == -1) {
        LOGE(LOG_ERROR, "error while forking");
        return -1;
    } else if (pid == 0) {
        while ((dup2(filedes[1], STDOUT_FILENO) == -1) && (errno == EINTR)) {}
        while ((dup2(filedes[1], STDERR_FILENO) == -1) && (errno == EINTR)) {}
        close(filedes[1]);
        close(filedes[0]);
        char *env[] = {NULL};
        execvpe(file, args, env);
        exit(1);
    }
    close(filedes[1]);
    return filedes[0];
}

kernel_info_t* utils_search_info(list *kernel_infos, const char *kernelname)
{
    kernel_info_t *info = NULL;
    if (kernel_infos == NULL) {
        LOGE(LOG_ERROR, "list is NULL.");
        return NULL;
    }
    LOGE(LOG_DBG(1), "searching for %s in %d entries", kernelname, kernel_infos->length);
    for (int i=0; i < kernel_infos->length; ++i) {
        if (list_at(kernel_infos, i, (void**)&info) != 0) {
            LOGE(LOG_ERROR, "no element at index %d", i);
        }
        if (strcmp(kernelname, info->name) == 0) {
            return info;
        }
    }
    return NULL;
}

int cpu_utils_is_local_connection(struct svc_req *rqstp)
{
    struct sockaddr_in remote_addr = {0};
    struct sockaddr_in local_addr = {0};
    struct hostent *hp;
    socklen_t sockaddr_len = sizeof(struct sockaddr_in);
    getpeername(rqstp->rq_xprt->xp_fd, &remote_addr, &sockaddr_len);
    getsockname(rqstp->rq_xprt->xp_fd, &local_addr, &sockaddr_len);
    //alternative (char*)inet_ntoa(((struct sockaddr_in*)rqstp->rq_xprt->xp_rtaddr.buf)->sin_addr)
    return (local_addr.sin_addr.s_addr == remote_addr.sin_addr.s_addr);
}


static int cpu_utils_read_pars(kernel_info_t *info, FILE* fdesc)
{
    static const char* attr_str[] = {"EIATTR_KPARAM_INFO",
        "EIATTR_CBANK_PARAM_SIZE",
        "EIATTR_PARAM_CBANK"};
    enum attr_t {KPARAM_INFO = 0,
        CBANK_PARAM_SIZE = 1,
        PARAM_CBANK = 2,
        ATTR_T_LAST}; // states for state machine
    char *line = NULL;
    size_t linelen = 0;
    int ret = 1;
    int read = 0;
    char key[32];
    char val[256] = {0};
    size_t val_len = 0;
    enum attr_t cur_attr = ATTR_T_LAST; // current state of state machine
    int consecutive_empty_lines = 0;
    info->param_num = 0;
    info->param_offsets = NULL;
    info->param_sizes = NULL;
    while (getline(&line, &linelen, fdesc) != -1) {
        memset(val, 0, 256);
        read = sscanf(line, "%31s %255c\n", key, val);
        val_len = strlen(val);
        if (val_len > 0) {
            val[strlen(val)-1] = '\0';
        }
        if (read == -1 || read == 0) {
            if (++consecutive_empty_lines >= 2) {
                break; //two empty line means there is no more info for this kernel
            } else {
                continue;
            }
        } else {
            consecutive_empty_lines = 0;
            if (read == 1) {
                continue; // some lines have no key-value pair.
                // We are not interested in those lines.
            }
        }
        if (strcmp(key, "Attribute:") == 0) { // state change
            LOG(LOG_DBG(3), "\"%s\", \"%s\"", key, val);
            cur_attr = ATTR_T_LAST;
            for (int i=0; i < ATTR_T_LAST; i++) {
                if (strcmp(val, attr_str[i]) == 0) {
                    LOG(LOG_DBG(3), "found %s", attr_str[i]);
                    cur_attr = i;
                }
            }
        } else if(strcmp(key, "Value:") == 0) {
            LOG(LOG_DBG(3), "\"%s\", \"%s\"", key, val);
            size_t buf;
            uint16_t ordinal, offset, size;
            switch(cur_attr) {
            case KPARAM_INFO:
                if (sscanf(val, "Index : 0x%*hx Ordinal : 0x%hx Offset : 0x%hx Size : 0x%hx\n", &ordinal, &offset, &size) != 3 ) {
                    LOGE(LOG_ERROR, "unexpected format of cuobjdump output");
                    goto cleanup;
                }
                if (ordinal >= info->param_num) {
                    info->param_offsets = realloc(
                                                  info->param_offsets,
                                                  (ordinal+1)*sizeof(uint16_t));
                    info->param_sizes = realloc(
                                                info->param_sizes,
                                                (ordinal+1)*sizeof(uint16_t));
                    info->param_num = ordinal+1;
                }
                info->param_offsets[ordinal] = offset;
                info->param_sizes[ordinal] = size;
                break;
            case CBANK_PARAM_SIZE:
                if (sscanf(val, "0x%lx", &info->param_size) != 1) {
                    LOGE(LOG_ERROR, "value has wrong format: key: %s, val: %s", key, val);
                    goto cleanup;
                }
                break;
            case PARAM_CBANK:
                if (sscanf(val, "0x%*x 0x%lx", &buf) != 1) {
                    LOGE(LOG_ERROR, "value has wrong format: key: %s, val: %s", key, val);
                    goto cleanup;
                }
                LOG(LOG_DBG(3), "found param address: %d", (uint16_t)(buf & 0xFFFF));
                break;
            default:
                break;
            }
        }


    }

    ret = 0;
 cleanup:
    free(line);
    return ret;
}

int cpu_utils_contains_kernel(const char *path)
{
    int ret = 1;
    char linktarget[PATH_MAX] = {0};
    char *args[] = {"/usr/local/cuda/bin/cuobjdump", "--dump-elf", NULL, NULL};
    int output;
    FILE *fdesc; //fd to read subcommands output from
    int child_exit = 0;
    char *line = NULL;
    size_t linelen;
    static const char nv_info_prefix[] = ".nv.info.";
    kernel_info_t *buf = NULL;
    char *kernelname;
    struct stat filestat = {0};

    if (stat(path, &filestat) != 0) {
        LOGE(LOG_ERROR, "stat on %s failed.", path);
        goto out;
    }

    if (S_ISLNK(filestat.st_mode)) {
        if (readlink("/proc/self/exe", linktarget, PATH_MAX) == PATH_MAX) {
            LOGE(LOG_ERROR, "executable path length is too long");
            goto out;
        }
        args[2] = linktarget;
    } else {
        args[2] = (char*)path;
    }
    LOG(LOG_DBG(1), "searching for kernels in \"%s\".", args[2]);

    if ( (output = cpu_utils_launch_child(args[0], args)) == -1) {
        LOGE(LOG_ERROR, "error while launching child.");
        goto out;
    }

    if ( (fdesc = fdopen(output, "r")) == NULL) {
        LOGE(LOG_ERROR, "erro while opening stream");
        goto cleanup;
    }

    if (getline(&line, &linelen, fdesc) != -1) {
        /*if (strncmp(line, nv_info_prefix, strlen(nv_info_prefix)) != 0) {
            // Line does not start with .nv.info. so continue searching.
            continue;
        }*/
        line[strlen(line)-1] = '\0';
        LOGE(LOG_DEBUG, "output: \"%s\"", line);
    }
    ret = 0;
    fclose(fdesc);
 cleanup:
    close(output);
    wait(&child_exit);
    LOG(LOG_DBG(1), "child exit code: %d", child_exit);
 out:
    free(line);
    return ret == 0 && child_exit == 0;
}

int cpu_utils_parameter_info(list *kernel_infos, char *path)
{
    int ret = 1;
    char linktarget[PATH_MAX] = {0};
    char *args[] = {"/usr/local/cuda/bin/cuobjdump", "--dump-elf", NULL, NULL};
    int output;
    FILE *fdesc; //fd to read subcommands output from
    int child_exit = 0;
    char *line = NULL;
    size_t linelen;
    static const char nv_info_prefix[] = ".nv.info.";
    kernel_info_t *buf = NULL;
    char *kernelname;
    struct stat filestat = {0};

    if (path == NULL) {
        LOGE(LOG_ERROR, "path is NULL.");
        goto out;
    }

    if (kernel_infos == NULL) {
        LOGE(LOG_ERROR, "list is NULL.");
        goto out;
    }

    if (stat(path, &filestat) != 0) {
        LOGE(LOG_ERROR, "stat on %s failed.", path);
        goto out;
    }

    if (S_ISLNK(filestat.st_mode) || strcmp(path, "/proc/self/exe") == 0) {
        if (readlink("/proc/self/exe", linktarget, PATH_MAX) == PATH_MAX) {
            LOGE(LOG_ERROR, "executable path length is too long");
            goto out;
        }
        args[2] = linktarget;
    } else {
        args[2] = path;
    }
    LOG(LOG_DBG(1), "searching for kernels in \"%s\".", args[2]);

    if ( (output = cpu_utils_launch_child(args[0], args)) == -1) {
        LOGE(LOG_ERROR, "error while launching child.");
        goto out;
    }

    if ( (fdesc = fdopen(output, "r")) == NULL) {
        LOGE(LOG_ERROR, "erro while opening stream");
        goto cleanup1;
    }

    while (getline(&line, &linelen, fdesc) != -1) {
        if (strncmp(line, nv_info_prefix, strlen(nv_info_prefix)) != 0) {
            // Line does not start with .nv.info. so continue searching.
            continue;
        }
        // Line starts with .nv.info.
        // Kernelname is line + strlen(nv_info_prefix)
        kernelname = line + strlen(nv_info_prefix);
        if (strlen(kernelname) == 0) {
            LOGE(LOG_ERROR, "found .nv.info section, but kernelname is empty");
            goto cleanup2;
        }

        if (list_append(kernel_infos, (void**)&buf) != 0) {
            LOGE(LOG_ERROR, "error on appending to list");
            goto cleanup2;
        }

        size_t buflen = strlen(kernelname);
        if ((buf->name = malloc(buflen)) == NULL) {
            LOGE(LOG_ERROR, "malloc failed");
            goto cleanup2;
        }
        //copy string and remove trailing \n
        strncpy(buf->name, kernelname, buflen-1);
        buf->name[buflen-1] = '\0';

        if (cpu_utils_read_pars(buf, fdesc) != 0) {
            LOGE(LOG_ERROR, "reading paramter infos failed.\n");
            goto cleanup2;
        }

        LOG(LOG_DEBUG, "found kernel \"%s\" [param_num: %d, param_size: %d]",
            buf->name, buf->param_num, buf->param_size);

    }

    if (ferror(fdesc) != 0) {
        LOGE(LOG_ERROR, "file descriptor shows an error");
        goto cleanup2;
    }

    ret = 0;
 cleanup2:
    fclose(fdesc);
 cleanup1:
    close(output);
    wait(&child_exit);
    LOG(LOG_DBG(1), "child exit code: %d", child_exit);
 out:
    free(line);
    return ret == 0 && child_exit == 0;
}

void kernel_infos_free(kernel_info_t *infos, size_t kernelnum)
{
    for (int i=0; i < kernelnum; ++i) {
        free(infos[i].name);
        free(infos[i].param_offsets);
        free(infos[i].param_sizes);
    }
}

void hexdump(const uint8_t* data, size_t size)
{
    size_t pos = 0;
    while (pos < size) {
        printf("%#05zx: ", pos);
        for (int i = 0; i < 16; i++) {
            if (pos + i < size) {
                printf("%02x", data[pos + i]);
            } else {
                printf("  ");
            }
            if (i % 4 == 3) {
                printf(" ");
            }
        }
        printf(" | ");
        for (int i = 0; i < 16; i++) {
            if (pos + i < size) {
                if (data[pos + i] >= 0x20 && data[pos + i] <= 0x7e) {
                    printf("%c", data[pos + i]);
                } else {
                    printf(".");
                }
            } else {
                printf(" ");
            }
        }
        printf("\n");
        pos += 16;
    }
}
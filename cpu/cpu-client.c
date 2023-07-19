#define _GNU_SOURCE
#include <cuda.h>
#include <driver_types.h>
#include <link.h>
#include <stdio.h>
#include <stdlib.h>

// For TCP socket
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include "cpu-common.h"
#include "cpu-libwrap.h"
#include "cpu-utils.h"
#include "cpu_rpc_prot.h"
#include "list.h"
#include "cpu-elf2.h"
#ifdef WITH_IB
#include "cpu-ib.h"
#endif // WITH_IB

// static const char* LIBCUDA_PATH = "/lib64/libcuda.so";
const char *LIBCUDA_PATH = "/usr/local/cuda/lib64/libcudart.so";

CLIENT *clnt = NULL;

list kernel_infos = { 0 };

char server[256];

INIT_SOCKTYPE
int connection_is_local = 0;
int shm_enabled = 1;
int initialized = 0;

#ifdef WITH_IB
int ib_device = 0;
#endif // WITH_IB

#ifdef WITH_API_CNT
extern void cpu_runtime_print_api_call_cnt(void);
#endif // WITH_API_CNT

static void rpc_connect(void)
{
    int isock;
    struct sockaddr_un sock_un = { 0 };
    struct sockaddr_in sock_in = { 0 };
    struct sockaddr_in local_addr = { 0 };
    struct hostent *hp;
    socklen_t sockaddr_len = sizeof(struct sockaddr_in);
    unsigned long prog = 0, vers = 0;

    char envvar[] = "REMOTE_GPU_ADDRESS";

    if (!getenv(envvar)) {
        LOG(LOG_ERROR,
            "Environment variable %s does not exist. It must contain the "
            "address where the server application is listening.",
            envvar);
        exit(1);
    }
    if (strncpy(server, getenv(envvar), 256) == NULL) {
        LOGE(LOG_ERROR, "strncpy failed.");
        exit(1);
    }
    LOG(LOG_INFO, "connection to host \"%s\"", server);

#ifdef WITH_IB

    if (getenv("IB_DEVICE_ID")) {
        ib_device = atoi(getenv("IB_DEVICE_ID"));
    }
    LOG(LOG_INFO, "Using IB device: %d.", ib_device);

#endif // WITH_IB

    prog = 99;
    vers = 1;
    const char *env_vers = getenv("CRICKET_RPCID");
    if (env_vers != NULL) {
        if (sscanf(env_vers, "%lu", &vers) != 1) {
            LOGE(LOG_ERROR, "error parsing CRICKET_RPCID");
            exit(1);
        }
    }

    char *cmd = NULL;
    if (cpu_utils_command(&cmd) != 0) {
        LOGE(LOG_ERROR, "error getting command");
    } else {
        LOG(LOG_DEBUG, "the command is \"%s\"", cmd);
    }
    free(cmd);

    LOGE(LOG_DEBUG, "using prog=%d, vers=%d", prog, vers);

    switch (socktype) {
    case UNIX:
        LOG(LOG_INFO, "connecting via UNIX...");
        isock = RPC_ANYSOCK;
        sock_un.sun_family = AF_UNIX;
        strcpy(sock_un.sun_path, CD_SOCKET_PATH);
        clnt = clntunix_create(&sock_un, prog, vers, &isock, 0, 0);
        connection_is_local = 1;
        break;
    case TCP:
        LOG(LOG_INFO, "connecting via TCP...");
        isock = RPC_ANYSOCK;
        sock_in.sin_family = AF_INET;
        sock_in.sin_port = 0;
        if ((hp = gethostbyname(server)) == 0) {
            LOGE(LOG_ERROR, "error resolving hostname: %s", server);
            exit(1);
        }
        sock_in.sin_addr = *(struct in_addr *)hp->h_addr;
        // inet_aton("137.226.133.199", &sock_in.sin_addr);

        clnt = clnttcp_create(&sock_in, prog, vers, &isock, 0, 0);
        getsockname(isock, &local_addr, &sockaddr_len);
        connection_is_local =
            (local_addr.sin_addr.s_addr == sock_in.sin_addr.s_addr);
        break;
    case UDP:
        /* From RPCEGEN documentation:
         * Warning: since UDP-based RPC messages can only hold up to 8 Kbytes
         * of encoded data, this transport cannot be used for procedures that
         * take large arguments or return huge results.
         * -> Sounds like UDP does not make sense for CUDA, because we need to
         *    be able to copy large memory chunks
         **/
        printf("UDP is not supported...\n");
        break;
    }

    if (clnt == NULL) {
        clnt_pcreateerror("[rpc] Error");
        exit(1);
    }
}

static void repair_connection(int signo)
{
    enum clnt_stat retval_1;
    int result_1;
    /*LOGE(LOG_INFO, "Trying connection...");
    char *printmessage_1_arg1 = "connection test";
    retval_1 = rpc_printmessage_1(printmessage_1_arg1, &result_1, clnt);
    printf("return:%d\n", result_1);
    if (retval_1 == RPC_SUCCESS) {
        LOG(LOG_INFO, "connection still okay.");
        return;
    }*/
    LOG(LOG_INFO, "connection dead. Reconnecting...");
    rpc_connect();
    LOG(LOG_INFO, "reconnected");
    retval_1 = cuda_device_synchronize_1(&result_1, clnt);
    if (retval_1 != RPC_SUCCESS) {
        LOGE(LOG_ERROR, "error calling cudaDeviceSynchronize");
    }
}

void __attribute__((constructor)) init_rpc(void)
{
    enum clnt_stat retval_1;
    int result_1;
    int_result result_2;
    char *printmessage_1_arg1 = "hello";

    LOG(LOG_DBG(1), "log level is %d", LOG_LEVEL);
    init_log(LOG_LEVEL, __FILE__);
    rpc_connect();

    initialized = 1;
    if (signal(SIGUSR1, repair_connection) == SIG_ERR) {
        LOGE(LOG_ERROR, "An error occurred while setting a signal handler.");
        exit(1);
    }

    retval_1 = rpc_printmessage_1(printmessage_1_arg1, &result_1, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror(clnt, "call failed");
    }

    if (list_init(&kernel_infos, sizeof(kernel_info_t)) != 0) {
        LOGE(LOG_ERROR, "list init failed.");
    }

    if (elf2_init() != 0) {
        LOGE(LOG_ERROR, "libelf init failed");
    }

    // if (cpu_utils_parameter_info(&kernel_infos, "/proc/self/exe") != 0) {
    //     LOG(LOG_ERROR, "error while getting parameter size. Check whether "
    //                    "cuobjdump binary is in PATH! Trying anyway (will only "
    //                    "work if there is no kernel in this binary)");
    // }
#ifdef WITH_IB
    if (ib_init(ib_device, server) != 0) {
        LOG(LOG_ERROR, "initilization of infiniband verbs failed.");
    }
#endif // WITH_IB
}
void __attribute__((destructor)) deinit_rpc(void)
{
    enum clnt_stat retval_1;
    int result;
    if (initialized) {
        retval_1 = rpc_deinit_1(&result, clnt);
        if (retval_1 != RPC_SUCCESS) {
            LOGE(LOG_ERROR, "call failed.");
        }
        kernel_infos_free(kernel_infos.elements, kernel_infos.length);
        list_free(&kernel_infos);
#ifdef WITH_API_CNT
        cpu_runtime_print_api_call_cnt();
#endif // WITH_API_CNT
    }

    if (clnt != NULL) {
        clnt_destroy(clnt);
    }
}


static void *(*dlopen_orig)(const char *, int) = NULL;
static int (*dlclose_orig)(void *) = NULL;
static void *dl_handle = NULL;

void *dlopen(const char *filename, int flag)
{
    void *ret = NULL;
    struct link_map *map;
    int has_kernel = 0;
    LOG(LOG_DBG(1), "intercepted dlopen(%s, %d)", filename, flag);

    if (filename == NULL) {
        return dlopen_orig(filename, flag);
    }

    if (dlopen_orig == NULL) {
        if ((dlopen_orig = dlsym(RTLD_NEXT, "dlopen")) == NULL) {
            LOGE(LOG_ERROR, "[dlopen] dlsym failed");
        }
    }

    static const char *replace_libs[] = {
        "libcuda.so.1",
        "libcuda.so",
        "libnvidia-ml.so.1",
        "libcudnn_cnn_infer.so.8"
    };
    static const size_t replace_libs_sz = sizeof(replace_libs) / sizeof(char *);
    if (filename != NULL) {
        for (size_t i=0; i != replace_libs_sz; ++i) {
            if (strcmp(filename, replace_libs[i]) == 0) {
                LOG(LOG_DEBUG, "replacing dlopen call to %s with cricket-client.so", filename);
                dl_handle = dlopen_orig("cricket-client.so", flag);
                if (clnt == NULL) {
                    LOGE(LOG_ERROR, "rpc seems to be uninitialized");
                }
                return dl_handle;
            }
        }
    }
    /* filename is NULL or not in replace_libs list */
    if ((ret = dlopen_orig(filename, flag)) == NULL) {
        LOGE(LOG_ERROR, "dlopen failed: ", dlerror());
    } else if (has_kernel) {
        dlinfo(ret, RTLD_DI_LINKMAP, &map);
        LOGE(LOG_DEBUG, "dlopen to  %p", map->l_addr);
    }
    return ret;
}

int dlclose(void *handle)
{
    if (handle == NULL) {
        LOGE(LOG_ERROR, "[dlclose] handle NULL");
        return -1;
    } else if (dlclose_orig == NULL) {
        if ((dlclose_orig = dlsym(RTLD_NEXT, "dlclose")) == NULL) {
            LOGE(LOG_ERROR, "[dlclose] dlsym failed");
        }
    }

    // Ignore dlclose call that would close this library
    if (dl_handle == handle) {
        LOGE(LOG_DEBUG, "[dlclose] ignore close");
        return 0;
    } else {
        return dlclose_orig(handle);
    }
}

void __cudaRegisterVar(void **fatCubinHandle, char *hostVar, char *deviceAddress,
                       const char *deviceName, int ext, size_t size, int constant,
                       int global)
{
    enum clnt_stat retval_1;
    int result;
    LOGE(LOG_DEBUG, "__cudaRegisterVar(fatCubinHandle=%p, hostVar=%p, deviceAddress=%p, "
           "deviceName=%s, ext=%d, size=%zu, constant=%d, global=%d)\n",
           fatCubinHandle, hostVar, deviceAddress, deviceName, ext, size, constant, global);
    retval_1 = rpc_register_var_1((ptr)fatCubinHandle, (ptr)hostVar, (ptr)deviceAddress, (char*)deviceName, ext, size, constant, global,
                                       &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        LOGE(LOG_ERROR, "call failed.");
    }
}

void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun,
                            char *deviceFun, const char *deviceName,
                            int thread_limit, uint3 *tid, uint3 *bid,
                            dim3 *bDim, dim3 *gDim, int *wSize)
{
    ptr_result result;
    enum clnt_stat retval_1;

    LOGE(LOG_DEBUG, "__cudaRegisterFunction(fatCubinHandle=%p, hostFun=%p, devFunc=%s, "
           "deviceName=%s, thread_limit=%d, tid=[%p], bid=[%p], bDim=[%p], "
           "gDim=[%p], wSize=%p)\n",
           fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid,
           bid, bDim, gDim, wSize);

    kernel_info_t *info = utils_search_info(&kernel_infos, (char *)deviceName);
    if (info == NULL) {
        LOGE(LOG_ERROR, "request to register unknown function: \"%s\"",
             deviceName);
        return;
    } else {
        LOGE(LOG_DEBUG, "request to register known function: \"%s\"",
             deviceName);
        retval_1 = rpc_register_function_1((ptr)fatCubinHandle, (ptr)hostFun,
                                           deviceFun, (char*)deviceName, thread_limit,
                                           &result, clnt);
        if (retval_1 != RPC_SUCCESS) {
            LOGE(LOG_ERROR, "call failed.");
            exit(1);
        }
        if (result.err != 0) {
            LOGE(LOG_ERROR, "error registering function: %d", result.err);
            exit(1);
        }
        info->host_fun = (void *)hostFun;
    }
}


void **__cudaRegisterFatBinary(void *fatCubin)
{
    void **result;
    int rpc_result;
    enum clnt_stat retval_1;
    size_t fatbin_size;
    LOGE(LOG_DEBUG, "__cudaRegisterFatBinary(fatCubin=%p)", fatCubin);

    mem_data rpc_fat = { .mem_data_len = 0, .mem_data_val = NULL };

    if (elf2_get_fatbin_info((struct fat_header *)fatCubin,
                                &kernel_infos,
                                (uint8_t **)&rpc_fat.mem_data_val,
                                &fatbin_size) != 0) {
        LOGE(LOG_ERROR, "error getting fatbin info");
        return NULL;
    }
    rpc_fat.mem_data_len = fatbin_size;

    // CUDA registers an atexit handler for fatbin cleanup that accesses
    // the fatbin data structure. Let's allocate some zeroes to avoid segfaults.
    result = (void**)calloc(1, 0x58);

    retval_1 = rpc_elf_load_1(rpc_fat, (ptr)result, &rpc_result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        LOGE(LOG_ERROR, "call failed.");
    }
    if (rpc_result != 0) {
        LOGE(LOG_ERROR, "error registering fatbin: %d", rpc_result);
        return NULL;
    }
    LOG(LOG_DEBUG, "fatbin loaded to %p", result);
    // we return a bunch of zeroes to avoid segfaults. The memory is
    // mapped by the modules resource 
    return result;
}

void __cudaUnregisterFatBinary(void **fatCubinHandle)
{  
    int result;
    enum clnt_stat retval_1;

    LOGE(LOG_DEBUG, "__cudaUnregisterFatBinary(fatCubinHandle=%p)",
         fatCubinHandle);

    if (fatCubinHandle == NULL) {
        LOGE(LOG_WARNING, "fatCubinHandle is NULL - so we have nothing to unload. (This is okay if this binary does not contain a kernel.)");
        return;
    }

    // retval_1 = rpc_elf_unload_1((ptr)fatCubinHandle, &result, clnt);
    // if (retval_1 != RPC_SUCCESS || result != 0) {
    //     LOGE(LOG_ERROR, "call failed.");
    // }
}

// void __cudaRegisterFatBinaryEnd(void **fatCubinHandle)
// {
//     int result;
//     enum clnt_stat retval_1;

//     //printf("__cudaRegisterFatBinaryEnd(fatCubinHandle=%p)\n",
//     fatCubinHandle);

//     retval_1 =
//     RPC_SUCCESS;//cuda_register_fat_binary_end_1((uint64_t)fatCubinHandle,
//     &result, clnt); if (retval_1 != RPC_SUCCESS) {
//         clnt_perror (clnt, "call failed");
//     }
// }
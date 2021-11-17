#include <stdlib.h>
#include <stdio.h>
#include <sys/socket.h>
#include <unistd.h> //unlink()
#include <signal.h> //sigaction
#include <sys/types.h>
#include <sys/stat.h>

#include "cpu-server.h"
#include "cpu_rpc_prot.h"
#include "cpu-common.h"
#include "cpu-utils.h"
#include "log.h"
#include "cpu-server-runtime.h"
#include "rpc/xdr.h"
#include "cr.h"
#ifdef WITH_IB
#include "cpu-ib.h"
#endif //WITH_IB

INIT_SOCKTYPE

int connection_is_local = 0;
int shm_enabled = 1;

#ifdef WITH_IB
    int ib_device = 0;
#endif //WITH_IB


unsigned long prog=0, vers=0;

extern void rpc_cd_prog_1(struct svc_req *rqstp, register SVCXPRT *transp);

void int_handler(int signal) {
    if (socktype == UNIX) {
        unlink(CD_SOCKET_PATH);
    }
    LOG(LOG_INFO, "have a nice day!\n");
    svc_exit();
}

bool_t rpc_printmessage_1_svc(char *argp, int *result, struct svc_req *rqstp)
{
    LOG(LOG_INFO, "string: \"%s\"\n", argp);
    *result = 42;
    return 1;
}

bool_t rpc_deinit_1_svc(int *result, struct svc_req *rqstp)
{
    LOG(LOG_INFO, "RPC deinit requested.");
    svc_exit();
    return 1;
}

int cricket_server_checkpoint(int dump_memory)
{
    int ret;
    struct stat path_stat = { 0 };
    const char *ckp_path = "ckp";
    LOG(LOG_INFO, "rpc_checkpoint requested.");

    if (!ckp_path) {
        LOGE(LOG_ERROR, "ckp_path is NULL");
        goto error;
    }
    if (stat(ckp_path, &path_stat) != 0) {
        LOG(LOG_DEBUG, "directory \"%s\" does not exist. Let's create it.", ckp_path);
        if (mkdir(ckp_path, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0) {
            LOGE(LOG_ERROR, "failed to create directory \"%s\"", ckp_path);
            goto error;
        }
    } else if (!S_ISDIR(path_stat.st_mode)) {
        LOG(LOG_ERROR, "file \"%s\" is not a directory", ckp_path);
        goto error;
    }
    
    if ((ret = server_runtime_checkpoint(ckp_path, dump_memory, prog, vers)) != 0) {
        LOGE(LOG_ERROR, "server_runtime_checkpoint returned %d", ret);
        goto error;
    }

    LOG(LOG_INFO, "checkpoint successfully created.");
    return 0;
 error:
    LOG(LOG_INFO, "checkpoint creation failed.");
    return 1;
}

static void signal_checkpoint(int signo)
{
    if (cricket_server_checkpoint(1) != 0) {
        LOGE(LOG_ERROR, "failed to create checkpoint");
    }
}

bool_t rpc_checkpoint_1_svc(int *result, struct svc_req *rqstp)
{
    int ret;
    if ((ret = cricket_server_checkpoint(1)) != 0) {
        LOGE(LOG_ERROR, "failed to create checkpoint");
    }
    return ret == 0;
}

/** implementation for CUDA_REGISTER_FUNCTION(ptr, str, str, str, int)
 *
 */
bool_t cuda_register_function_1_svc(ptr fatCubinHandle, ptr hostFun, char* deviceFun, char* deviceName, int thread_limit, int* result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "cudaRegisterFunction(%p, %p, %s, %s, %d)", fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit);
    *result = 0;
    return 1;
}

void cricket_main_hash(char* app_command)
{
    cricket_main(app_command, 0, 0);
}

void cricket_main_static(size_t prog_num, size_t vers_num)
{
    cricket_main("", prog_num, vers_num);
}

void cricket_main(char* app_command, size_t prog_num, size_t vers_num)
{
    register SVCXPRT *transp;

    int protocol = 0;
    int restore = 0;
    struct sigaction act;
    char *command = NULL;
    act.sa_handler = int_handler;
    sigaction(SIGINT, &act, NULL);

    init_log(LOG_LEVEL, __FILE__);

    #ifdef WITH_IB
    char client[256];
    char envvar[] = "CLIENT_ADDRESS";

    if(!getenv(envvar)) {
        LOG(LOG_ERROR, "Environment variable %s does not exist. For memory transports using InfiniBand it must contain the address of the client.", envvar);
        exit(1);
    }
    if(strncpy(client, getenv(envvar), 256) == NULL) {
        LOGE(LOG_ERROR, "strncpy failed.");
        exit(1);
    }
    LOG(LOG_INFO, "connection to client \"%s\"", client);

    if(getenv("IB_DEVICE_ID")) {
        ib_device = atoi(getenv("IB_DEVICE_ID"));
    }
    LOG(LOG_INFO, "Using IB device: %d.", ib_device);

    #endif //WITH_IB


    if (getenv("CRICKET_DISABLE_RPC")) {
        LOG(LOG_INFO, "RPC server was disable by setting CRICKET_DISABLE_RPC");
        return;
    }
    if (getenv("CRICKET_RESTORE")) {
        LOG(LOG_INFO, "restoring previous state was enabled by setting CRICKET_RESTORE");
            restore = 1;
    }

    if (cpu_utils_command(&command) != 0) {
        LOG(LOG_WARNING, "could not retrieve command name. This might prevent starting CUDA applications");
    } else {
        LOG(LOG_DEBUG, "the command is '%s'", command);
        //This is a workaround to make LD_PRELOAD work under GDB supervision
        const char *cmp = "cudbgprocess";
        if (strncmp(command, cmp, strlen(cmp)) == 0) {
            LOG(LOG_DEBUG, "skipping RPC server");
            return;
        }
    }

    if (restore == 1) {
        if (cr_restore_rpc_id("ckp", &prog, &vers) != 0) {
            LOGE(LOG_ERROR, "error while restoring rpc id");
        }
    } else {
        if (prog_num == 0) {
            if (cpu_utils_md5hash(app_command, &prog, &vers) != 0) {
                LOGE(LOG_ERROR, "error while creating binary checksum");
                exit(0);
            }
        }
        else {
            prog = prog_num;
            vers = vers_num;
        }
    }

    LOGE(LOG_DEBUG, "using prog=%d, vers=%d, derived from \"%s\"", prog, vers, app_command);


    switch (socktype) {
    case UNIX:
        LOG(LOG_INFO, "using UNIX...");
        transp = svcunix_create(RPC_ANYSOCK, 0, 0, CD_SOCKET_PATH);
        if (transp == NULL) {
            LOGE(LOG_ERROR, "cannot create service.");
            exit(1);
        }
        connection_is_local = 1;
        break;
    case TCP:
        LOG(LOG_INFO, "using TCP...");
        transp = svctcp_create(RPC_ANYSOCK, 0, 0);
        if (transp == NULL) {
            LOGE(LOG_ERROR, "cannot create service.");
            exit(1);
        }
        pmap_unset(prog, vers);
        LOG(LOG_INFO, "listening on port %d", transp->xp_port);
        protocol = IPPROTO_TCP;
        break;
    case UDP:
        /* From RPCGEN documentation:
         * Warning: since UDP-based RPC messages can only hold up to 8 Kbytes
         * of encoded data, this transport cannot be used for procedures that
         * take large arguments or return huge results.
         * -> Sounds like UDP does not make sense for CUDA, because we need to
         *    be able to copy large memory chunks
         **/
        LOG(LOG_INFO, "UDP is not supported...");
        break;
    }

    if (!svc_register(transp, prog, vers, rpc_cd_prog_1, protocol)) {
        LOGE(LOG_ERROR, "unable to register (RPC_PROG_PROG, RPC_PROG_VERS).");
        exit(1);
    }

    /* Call CUDA initialization function (usually called by __libc_init_main())
     * Address of "_ZL24__sti____cudaRegisterAllv" in static symbol table is e.g. 0x4016c8
     */
    void (*cudaRegisterAllv)(void) =
        (void(*)(void)) cricketd_utils_symbol_address("_ZL24__sti____cudaRegisterAllv");
    LOG(LOG_INFO, "found CUDA initialization function at %p", cudaRegisterAllv);
    if (cudaRegisterAllv == NULL) {
        LOGE(LOG_WARNING, "could not find cudaRegisterAllv initialization function in cubin. Kernels cannot be launched without it!");
    } else {
        cudaRegisterAllv();
    }

    if (server_runtime_init(restore) != 0) {
        LOGE(LOG_ERROR, "initializing server_runtime failed.");
        exit(1);
    }

#ifdef WITH_IB

    if (ib_init(ib_device, client) != 0) {
        LOG(LOG_ERROR, "initilization of infiniband verbs failed.");
    }
    
#endif // WITH_IB


    if (signal(SIGUSR1, signal_checkpoint) == SIG_ERR) {
        LOGE(LOG_ERROR, "An error occurred while setting a signal handler.");
        exit(1);
    }

    LOG(LOG_INFO, "waiting for RPC requests...");

    svc_run();

    server_runtime_deinit();
    LOG(LOG_DEBUG, "svc_run returned. Cleaning up.");
    pmap_unset(prog, vers);
    svc_destroy(transp);
    unlink(CD_SOCKET_PATH);
    LOG(LOG_DEBUG, "have a nice day!");
    exit(0);
}

int rpc_cd_prog_1_freeresult (SVCXPRT * a, xdrproc_t b , caddr_t c)
{
    if (b == (xdrproc_t) xdr_str_result) {
        str_result *res = (str_result*)c;
        if (res->err == 0) {
            free( res->str_result_u.str);
        }
    }
    else if (b == (xdrproc_t) xdr_mem_result) {
        mem_result *res = (mem_result*)c;
        if (res->err == 0) {
            free( (void*)res->mem_result_u.data.mem_data_val);
        }
    }
    return 1;
}


#include <stdlib.h>
#include <stdio.h>
#include <sys/socket.h>
#include <unistd.h> //unlink()
#include <signal.h> //sigaction

#include "cpu_rpc_prot.h"
#include "cpu-common.h"
#include "cpu-utils.h"
#include "log.h"

INIT_SOCKTYPE

extern void rpc_cd_prog_1(struct svc_req *rqstp, register SVCXPRT *transp);

void int_handler(int signal) {
    if (socktype == UNIX) {
        unlink(CD_SOCKET_PATH);
    }
    LOG(LOG_INFO, "have a nice day!\n");
    exit(0);
}

bool_t printmessage_1_svc(char *argp, int *result, struct svc_req *rqstp)
{
    LOG(LOG_INFO, "string: \"%s\"\n", argp);
    *result = 42;
    return 1;
}


/* shared object constructor; executes before main and thus hijacks main program */
void __attribute__ ((constructor)) cricketd_main(void)
{
    register SVCXPRT *transp;

    int protocol = 0;
    struct sigaction act;
    act.sa_handler = int_handler;
    sigaction(SIGINT, &act, NULL);

    init_log(LOG_DEBUG, __FILE__);

    switch (socktype) {
    case UNIX:
        LOG(LOG_INFO, "using UNIX...\n");
        transp = svcunix_create(RPC_ANYSOCK, 0, 0, CD_SOCKET_PATH);
        if (transp == NULL) {
            LOGE(LOG_ERROR, "cannot create service.");
            exit(1);
        } 
        break;
    case TCP:
        LOG(LOG_INFO, "using TCP...\n");
        transp = svctcp_create(RPC_ANYSOCK, 0, 0);
        if (transp == NULL) {
            LOGE(LOG_ERROR, "cannot create service.");
            exit(1);
        } 
        pmap_unset(RPC_CD_PROG, RPC_CD_VERS); 
        LOG(LOG_INFO, "listening on port %d\n", transp->xp_port);
        protocol = IPPROTO_TCP;
        break;
    case UDP:
        /* From RPCEGEN documentation: 
         * Warning: since UDP-based RPC messages can only hold up to 8 Kbytes
         * of encoded data, this transport cannot be used for procedures that 
         * take large arguments or return huge results. 
         * -> Sounds like UDP does not make sense for CUDA, because we need to
         *    be able to copy large memory chunks
         **/
        LOG(LOG_INFO, "UDP is not supported...\n");
        break;
    }

    if (!svc_register(transp, RPC_CD_PROG, RPC_CD_VERS, rpc_cd_prog_1, protocol)) {
        fprintf (stderr, "%s", "unable to register (RPC_PROG_PROG, RPC_PROG_VERS).");
        exit(1);
    }

    /* Call CUDA initialization function (usually called by __libc_init_main())
     * Address of "_ZL24__sti____cudaRegisterAllv" in static symbol table is e.g. 0x4016c8
     */
    void (*cudaRegisterAllv)(void) =
        (void(*)(void)) cricketd_utils_symbol_address("_ZL24__sti____cudaRegisterAllv");
    LOG(LOG_INFO, "found CUDA initialization function at %p\n", cudaRegisterAllv);
    if (cudaRegisterAllv == NULL) {
        LOGE(LOG_ERROR, "cricketd: error: could not find cudaRegisterAllv initialization function in cubin. I cannot operate without it.\n");
        exit(1);
    }
    cudaRegisterAllv();

    LOG(LOG_INFO, "waiting for RPC requests...\n");

    svc_run ();
    fprintf (stderr, "%s", "svc_run returned");
    unlink(CD_SOCKET_PATH);
    exit(0);
}

int rpc_cd_prog_1_freeresult (SVCXPRT * a, xdrproc_t b , caddr_t c)
{
    if (b == (xdrproc_t) xdr_str_result) {
        free( ((str_result*)c)->str_result_u.str);
    }
}

#include <stdlib.h>
#include <stdio.h>

#include "cd_rpc_prot.h"
#include "cd_common.h"

int main(int argc, char* argv[])
{
	CLIENT *clnt;
	enum clnt_stat retval_1;
	int result_1;
    int_result result_2;
	char *printmessage_1_arg1 = "hello";
    struct sockaddr_un sock = {.sun_family = AF_UNIX,
                               .sun_path = CD_SOCKET_PATH};
    int isock = RPC_ANYSOCK;
    clnt = clntunix_create(&sock, RPC_CD_PROG, RPC_CD_VERS, &isock, 0, 0);
	if (clnt == NULL) {
        printf("error\n");
		exit (1);
	}

	retval_1 = printmessage_1(printmessage_1_arg1, &result_1, clnt);
    printf("return:%d\n", result_1);
	if (retval_1 != RPC_SUCCESS) {
		clnt_perror (clnt, "call failed");
	}

    retval_1 = rpc_cuinit_1(0, &result_1, clnt);
    printf("return %d\n", result_1);
	if (retval_1 != RPC_SUCCESS) {
		clnt_perror (clnt, "call failed");
	}


    retval_1 = rpc_cudevicegetcount_1(&result_2, clnt);
    printf("error %d, result %d\n", result_2.err, result_2.int_result_u.data);
	if (retval_1 != RPC_SUCCESS) {
		clnt_perror (clnt, "call failed");
	}

	clnt_destroy (clnt);
    return 0;
}

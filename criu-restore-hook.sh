#!/bin/bash

case "$CRTOOLS_SCRIPT_ACTION" in
	"pre-resume" )
        client_pid=$(cat client_pid)
        #client_pid=$(pgrep matrixMul)
        echo "sending USR1 to $CUDA_APP_NAME with pid $client_pid"
        kill -s USR1 $client_pid
		exit $?
		;;
esac

exit 0

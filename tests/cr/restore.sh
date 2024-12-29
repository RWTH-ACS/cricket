#!/bin/bash

CUDA_APP=${CRICKET_PATH}/tests/samples/samples-bin/nbody.compressed.sample
export CUDA_APP_NAME=$(basename nbody)

CRICKET_CLIENT=${CRICKET_PATH}/cpu/cricket-client.so
CRICKET_SERVER=${CRICKET_PATH}/cpu/cricket-rpc-server
CRICKET_SERVER_BIN=$(basename ${CRICKET_SERVER})
CRIU=criu

export REMOTE_GPU_ADDRESS=ghost.acs-lab.eonerc.rwth-aachen.de
export CUDA_VISIBLE_DEVICES=3

sudo killall ${CUDA_APP_NAME}
sudo killall ${CRICKET_SERVER_BIN}
sudo killall criu

CRICKET_RESTORE=yes ${CRICKET_SERVER} &
server_pid=$!
echo "server pid:" $server_pid
sleep 3

sudo ${CRIU} restore -vvvvvv --tcp-established --shell-job --images-dir ./criu-ckp -W ./criu-ckp --action-script $(pwd)/criu-restore-hook.sh -d -o restore.log

sleep 10

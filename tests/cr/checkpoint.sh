#!/bin/bash

CUDA_APP=${CRICKET_PATH}/tests/samples/samples-bin/nbody.compressed.sample
export CUDA_APP_NAME=$(basename nbody)
CUDA_APP_ARGS="-benchmark -numbodies=25600"


CRICKET_CLIENT=${CRICKET_PATH}/cpu/cricket-client.so
CRICKET_SERVER=${CRICKET_PATH}/cpu/cricket-rpc-server
CRICKET_SERVER_BIN=$(basename ${CRICKET_SERVER})
CRIU=criu

export REMOTE_GPU_ADDRESS=ghost.acs-lab.eonerc.rwth-aachen.de
export CUDA_VISIBLE_DEVICES=3

rm -rf /tmp/cricket-ckp/*
rm -rf criu-ckp/*
rm -rf ckp/*
mkdir -p criu-ckp
sudo killall ${CUDA_APP_NAME}
sudo killall ${CRICKET_SERVER_BIN}

${CRICKET_SERVER}&
server_pid=$!
echo "server pid:" $server_pid
sleep 1

LD_PRELOAD=${CRICKET_CLIENT} ${CUDA_APP} ${CUDA_APP_ARGS}&
client_pid=$!
echo "client pid:" $client_pid

sleep 2
#sleep 2.95

kill -s USR1 ${server_pid} && \
echo $client_pid > criu-ckp/client_pid && \
sudo ${CRIU} dump -vvvvvv -t $client_pid --tcp-established --shell-job --images-dir ./criu-ckp -W ./criu-ckp -o dump.log

sleep 8

#kill $server_pid


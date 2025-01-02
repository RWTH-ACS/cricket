#!/bin/bash

# tested with CRIU commit ab288c35cb3ac4988151bcec61d44f54a8052dd7

CUDA_APP=${CRICKET_PATH}/tests/samples/samples-bin/nbody.compressed.sample
export CUDA_APP_NAME=$(basename nbody)
CUDA_APP_ARGS="-compare -numbodies=12800"


CRICKET_CLIENT=${CRICKET_PATH}/cpu/cricket-client.so
CRICKET_SERVER=${CRICKET_PATH}/cpu/cricket-rpc-server
CRICKET_SERVER_BIN=$(basename ${CRICKET_SERVER})
CRIU=criu

export REMOTE_GPU_ADDRESS=ghost.acs-lab.eonerc.rwth-aachen.de
export CUDA_VISIBLE_DEVICES=3

# clean up previous runs
rm -rf /tmp/cricket-ckp/*
rm -rf criu-ckp/*
rm -rf ckp/*
mkdir -p criu-ckp
sudo killall ${CUDA_APP_NAME}
sudo killall ${CRICKET_SERVER_BIN}

# start cricket server
${CRICKET_SERVER}&
server_pid=$!
echo "server pid:" $server_pid

# wait until cricket server is up
sleep 1

# start app
LD_PRELOAD=${CRICKET_CLIENT} ${CUDA_APP} ${CUDA_APP_ARGS}&
client_pid=$!
echo "client pid:" $client_pid

# wait until app is doing something interesting
sleep 1

# create the checkpoint by sending USR1 to the cricket server and dumping the client using CRIU
kill -s USR1 ${server_pid} && \
echo $client_pid > criu-ckp/client_pid && \
sudo ${CRIU} dump -vvvvvv -t $client_pid --tcp-established --shell-job --images-dir ./criu-ckp -W ./criu-ckp -o dump.log

# wait until the checkpoint is done
sleep 8


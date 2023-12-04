#!/bin/bash

CRICKET_PATH=$(pwd)/../..
CRICKET_BIN=${CRICKET_PATH}/gpu/cricket

if [ -z $1 ]; then
    CUDA_APP=${CRICKET_PATH}/bin/tests/kernel.testapp
elif [ -f $1 ]; then
    CUDA_APP=$1
elif [ -f ${CRICKET_PATH}/bin/tests/$1 ]; then
    CUDA_APP=${CRICKET_PATH}/bin/tests/$1
else
    echo "file $1 does not exist"
    exit 1
fi
echo "using $CUDA_APP"

CUDA_APP_NAME="$(basename -- $CUDA_APP)"
CRICKET_CLIENT=${CRICKET_PATH}/cpu/cricket-client.so
CRICKET_SERVER=${CRICKET_PATH}/cpu/cricket-server.so
CRIU=/home/eiling/tmp/criu/criu/criu

export REMOTE_GPU_ADDRESS=localhost
export CUDA_VISIBLE_DEVICES=0

CRICKET_CKP_DIR=/tmp/cricket-ckp
CRIU_CKP_DIR=/tmp/criu-ckp

rm -rf ${CRICKET_CKP_DIR}/*
rm -rf ${CRIU_CKP_DIR}/*
mkdir -p ${CRIU_CKP_DIR}
sudo killall ${CUDA_APP_NAME}

${CRICKET_BIN} start ${CUDA_APP} &

sleep 1
server_pid=$(pgrep ${CUDA_APP_NAME})
echo "server pid:" $server_pid

LD_PRELOAD=${CRICKET_CLIENT} ${CUDA_APP} &
client_pid=$!
echo "client pid:" $client_pid

sleep 5

sudo ${CRIU} dump -vvvvvv -t $client_pid --tcp-established --shell-job --images-dir ${CRIU_CKP_DIR} -W ${CRIU_CKP_DIR} -o dump.log

echo $client_pid > ${CRIU_CKP_DIR}/client_pid

${CRICKET_BIN} checkpoint $server_pid

sleep 2

kill $server_pid

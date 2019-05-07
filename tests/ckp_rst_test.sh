#!/bin/bash

# Use only P4 GPU
export CUDA_VISIBLE_DEVICES="1"

CRICKET=../cricket
TEST_PATH=/home/eiling/projects/cricket/tests
TEST_BIN=test_kernel
CKP_PATH=/home/eiling/tmp/cricket-ckp

mkdir -p $CKP_PATH
rm $CKP_PATH/*

$CRICKET start $TEST_PATH/$TEST_BIN &

sleep 2

$CRICKET checkpoint `pgrep $TEST_BIN`

sleep 7

$CRICKET restore $TEST_PATH/$TEST_BIN



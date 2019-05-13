#!/bin/sh

# Use only P4 GPU
export CUDA_VISIBLE_DEVICES="1"

TEST_PATH=$(dirname $(readlink -f "$0"))
TEST_BIN=test_kernel
CRICKET_EXEC=${TEST_PATH}/../cricket
CKP_PATH=$(mktemp -d -t cricket_ckp.XXXXXXX)


$CRICKET_EXEC -s $TEST_PATH/$TEST_BIN
# $TEST_PATH/$TEST_BIN &

sleep 1.5

TEST_PID=$(pgrep $TEST_BIN)
echo "$CRICKET_EXEC -c $TEST_PID -d $CKP_PATH "
$CRICKET_EXEC -c $TEST_PID -d $CKP_PATH

sleep 4
echo ""
echo ""
echo ""
echo ""
echo "Restoring now"

echo "$CRICKET_EXEC -r $TEST_PATH/$TEST_BIN -d $CKP_PATH"
$CRICKET_EXEC -r $TEST_PATH/$TEST_BIN -d $CKP_PATH

rm  -r $CKP_PATH

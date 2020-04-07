#!/bin/bash

# Parameters:
# $2: number of iterations (default: 1)
if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters"
    echo "    usage:"
    echo "    avg.sh <file> <iterations>"
    exit 1;
else
    iterations="$2";
fi



echo "Executing $iterations times:";
counter=0;
until [ $counter -eq $iterations ]; do
    echo "($counter/$iterations)";

    LD_PRELOAD=/home/eiling/projects/libtirpc/install/usr/lib/libtirpc.so.3:/home/eiling/projects/cricket/cpu/cricket-server.so /home/eiling/gpu-benchmark/10.2/1_Utilities/bandwidthTest/bandwidthTest &

    sleep 1

    ssh eiling@epyc4 "LD_PRELOAD=/home/eiling/projects/libtirpc/install/usr/lib/libtirpc.so.3:/home/eiling/projects/cricket/cpu/cricket-client.so /home/eiling/gpu-benchmark/10.2/1_Utilities/bandwidthTest/bandwidthTest --memory=pageable | tee -a $1"
    #LD_PRELOAD=/home/eiling/projects/libtirpc/install/usr/lib/libtirpc.so.3:/home/eiling/projects/cricket/cpu/cricket-client.so /home/eiling/gpu-benchmark/10.2/1_Utilities/bandwidthTest/bandwidthTest --memory=pinned | tee -a $1

    kill -2 $(pgrep bandwidthTest)
#/home/eiling/gpu-benchmark/10.2/1_Utilities/bandwidthTest/bandwidthTest --memory=pinned | tee -a $1
#/home/eiling/gpu-benchmark/10.2/1_Utilities/bandwidthTest/bandwidthTest --memory=pageable | tee -a $1
    if [ $? -ne 0 ]; then
        echo "proccess encountered error";
        exit 1;
    fi

    sleep 1;

    let counter+=1
done

exit 0;

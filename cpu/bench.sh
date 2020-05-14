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

#cd /home/eiling/projects/dpsim

echo "Executing $iterations times:";
counter=0;
until [ $counter -eq $iterations ]; do
    echo "($counter/$iterations)";

#CUDA_VISIBLE_DEVICES=1 LD_PRELOAD=/home/eiling/projects/libtirpc/install/usr/lib/libtirpc.so.3:/home/eiling/projects/cricket/cpu/cricket-server.so /home/eiling/gpu-benchmark/10.2/5_Simulations/nbody/nbody &
#CUDA_VISIBLE_DEVICES=1 LD_PRELOAD=/home/eiling/projects/libtirpc/install/usr/lib/libtirpc.so.3:/home/eiling/projects/cricket/cpu/cricket-server.so /home/eiling/gpu-benchmark/10.2/0_Simple/matrixMul/matrixMul &
#CUDA_VISIBLE_DEVICES=1 LD_PRELOAD=/home/eiling/projects/libtirpc/install/usr/lib/libtirpc.so.3:/home/eiling/projects/cricket/cpu/cricket-server.so /home/eiling/projects/dpsim/build/Examples/Cxx/WSCC_9bus_mult_coupled &
CUDA_VISIBLE_DEVICES=3 /usr/bin/time -f "%U %S %e" /bin/bash -c "/home/eiling/gpu-benchmark/10.2/0_Simple/matrixMul/matrixMul" 2>&1 | tee -a $1
#CUDA_VISIBLE_DEVICES=1 /usr/bin/time -f "%U %S %e" /home/eiling/gpu-benchmark/10.2/5_Simulations/nbody/nbody -benchmark -numbodies=256000 2>&1 | tee -a $1
#CUDA_VISIBLE_DEVICES=1 /usr/bin/time -f "%U %S %e" build/Examples/Cxx/WSCC_9bus_mult_coupled -o copies=6 2>&1 | tee -a $1


    sleep 2

#ssh eiling@epyc4 '/usr/bin/time -f "%U %S %e" /bin/bash -c "cd /home/eiling/projects/dpsim && PATH=/home/eiling/projects/cricket/gpu:$PATH LD_PRELOAD=/home/eiling/projects/libtirpc/install/usr/lib/libtirpc.so.3:/home/eiling/projects/cricket/cpu/cricket-client.so /home/eiling/projects/dpsim/build/Examples/Cxx/WSCC_9bus_mult_coupled -o copies=6"' 2>&1 | tee -a $1
#/usr/bin/time -f "%U %S %e" /bin/bash -c "cd /home/eiling/projects/dpsim && PATH=/home/eiling/projects/cricket/gpu:$PATH LD_PRELOAD=/home/eiling/projects/libtirpc/install/usr/lib/libtirpc.so.3:/home/eiling/projects/cricket/cpu/cricket-client.so /home/eiling/projects/dpsim/build/Examples/Cxx/WSCC_9bus_mult_coupled -o copies=6" 2>&1 | tee -a $1
#ssh eiling@epyc4 '/usr/bin/time -f "%U %S %e" /bin/bash -c "PATH=/home/eiling/projects/cricket/gpu:$PATH LD_PRELOAD=/home/eiling/projects/libtirpc/install/usr/lib/libtirpc.so.3:/home/eiling/projects/cricket/cpu/cricket-client.so /home/eiling/gpu-benchmark/10.2/5_Simulations/nbody/nbody -benchmark -numbodies=256000"' 2>&1 | tee -a $1
#ssh eiling@epyc4 '/usr/bin/time -f "%U %S %e" /bin/bash -c "PATH=/home/eiling/projects/cricket/gpu:$PATH LD_PRELOAD=/home/eiling/projects/libtirpc/install/usr/lib/libtirpc.so.3:/home/eiling/projects/cricket/cpu/cricket-client.so /home/eiling/gpu-benchmark/10.2/0_Simple/matrixMul/matrixMul"' 2>&1 | tee -a $1
#/usr/bin/time -f "%U %S %e" /bin/bash -c "PATH=/home/eiling/projects/cricket/gpu:$PATH LD_PRELOAD=/home/eiling/projects/libtirpc/install/usr/lib/libtirpc.so.3:/home/eiling/projects/cricket/cpu/cricket-client.so /home/eiling/gpu-benchmark/10.2/0_Simple/matrixMul/matrixMul" 2>&1 | tee -a $1
#/usr/bin/time -f "%U %S %e" /bin/bash -c "PATH=/home/eiling/projects/cricket/gpu:$PATH LD_PRELOAD=/home/eiling/projects/libtirpc/install/usr/lib/libtirpc.so.3:/home/eiling/projects/cricket/cpu/cricket-client.so /home/eiling/gpu-benchmark/10.2/5_Simulations/nbody/nbody -benchmark -numbodies=256000" 2>&1 | tee -a $1
    #LD_PRELOAD=/home/eiling/projects/libtirpc/install/usr/lib/libtirpc.so.3:/home/eiling/projects/cricket/cpu/cricket-client.so /home/eiling/gpu-benchmark/10.2/1_Utilities/bandwidthTest/bandwidthTest --memory=pinned | tee -a $1

#/home/eiling/gpu-benchmark/10.2/1_Utilities/bandwidthTest/bandwidthTest --memory=pinned | tee -a $1
#/home/eiling/gpu-benchmark/10.2/1_Utilities/bandwidthTest/bandwidthTest --memory=pageable | tee -a $1
    if [ $? -ne 0 ]; then
        echo "proccess encountered error";
        exit 1;
    fi
#kill -2 $(pgrep nbody)

    sleep 1;

    let counter+=1
done

exit 0;

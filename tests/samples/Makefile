CC = gcc
LD = gcc
CFLAGS = -Wall -std=gnu99 -g -ggdb
SAMPLES = samples-bin/matrixMul.compressed.sample \
		  samples-bin/matrixMul.uncompressed.sample \
		  samples-bin/nbody.uncompressed.sample \
		  samples-bin/nbody.compressed.sample \
		  samples-bin/bandwidthTest.sample \
		  samples-bin/mnistCUDNN.sample

CUDA_PATH = /usr/local/cuda
SMS = 75 60
CUDA_SAMPLES_RELEASE ?= 12.1
CUDA_SAMPLES_URL = https://github.com/NVIDIA/cuda-samples/archive/refs/tags/v${CUDA_SAMPLES_RELEASE}.tar.gz
CUDNN_SAMPLES_URL = https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/libcudnn8-samples-8.9.2.26-1.cuda12.1.x86_64.rpm

PWD = $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

.PHONY: all clean distclean

all : $(SAMPLES)

samples:
	mkdir -p $@
	wget ${CUDA_SAMPLES_URL} -O - | tar -xz --strip-components=1 -C $@

cudnn-samples:
	mkdir -p $@
	wget ${CUDNN_SAMPLES_URL} -O - | rpm2archive - | tar zxf - --strip-components=4 -C $@

samples-bin:
	mkdir -p $@

samples-bin/data:
	cp -R cudnn-samples/mnistCUDNN/data $@

samples-bin/mnistCUDNN.sample : cudnn-samples samples-bin samples-bin/data
	make -C cudnn-samples/mnistCUDNN \
		clean
	make -C cudnn-samples/mnistCUDNN \
		NVCCFLAGS="-cudart shared --no-compress -G" \
		SMS="${SMS}" \
		CUDA_PATH=${CUDA_PATH} \
		DEBUG=1
	cp cudnn-samples/mnistCUDNN/mnistCUDNN $@

samples-bin/nbody.uncompressed.sample : samples samples-bin
	make -C samples/Samples/5_Domain_Specific/nbody \
		clean
	make -C samples/Samples/5_Domain_Specific/nbody \
		NVCCFLAGS="-cudart shared --no-compress -g -G" \
		SMS="${SMS}" \
		CPATH="samples/Common" \
		CUDA_PATH=${CUDA_PATH}
	cp samples/Samples/5_Domain_Specific/nbody/nbody $@

samples-bin/nbody.compressed.sample : samples samples-bin
	make -C samples/Samples/5_Domain_Specific/nbody \
		clean
	make -C samples/Samples/5_Domain_Specific/nbody \
		NVCCFLAGS="-cudart shared -Xfatbin --compress-all -g -G" \
		SMS="${SMS}" \
		CPATH="samples/Common" \
		CUDA_PATH=${CUDA_PATH}
	cp samples/Samples/5_Domain_Specific/nbody/nbody $@

samples-bin/matrixMul.compressed.sample : samples samples-bin
	make -C samples/Samples/0_Introduction/matrixMul \
		clean
	make -C samples/Samples/0_Introduction/matrixMul \
		NVCCFLAGS="-cudart shared -Xfatbin --compress-all" \
		SMS="${SMS}" \
		CPATH="samples/Common" \
		CUDA_PATH=${CUDA_PATH}
	cp samples/Samples/0_Introduction/matrixMul/matrixMul $@

samples-bin/matrixMul.uncompressed.sample : samples samples-bin
	make -C samples/Samples/0_Introduction/matrixMul \
		clean
	make -C samples/Samples/0_Introduction/matrixMul \
		NVCCFLAGS="-cudart shared --no-compress" \
		SMS="${SMS}" \
		CPATH="samples/Common" \
		CUDA_PATH=${CUDA_PATH}
	cp samples/Samples/0_Introduction/matrixMul/matrixMul $@

samples-bin/bandwidthTest.sample : samples samples-bin
	make -C samples/Samples/1_Utilities/bandwidthTest \
		clean
	make -C samples/Samples/1_Utilities/bandwidthTest \
		NVCCFLAGS="-cudart shared --no-compress" \
		SMS="${SMS}" \
		CPATH="samples/Common" \
		CUDA_PATH=${CUDA_PATH}
	cp samples/Samples/1_Utilities/bandwidthTest/bandwidthTest $@

clean :
	rm -rf samples-bin

distclean : clean
	rm -rf samples
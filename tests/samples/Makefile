CC = gcc
LD = gcc
CFLAGS = -Wall -std=gnu99
ARCH = sm_61
CUDA_DIR = /usr/local/cuda

PWD = $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

.PHONY: all clean distclean

all : matrixMul/matrixMul bandwidthTest/bandwidthTest 

matrixMul :
	mkdir -p $(PWD)/matrixMul
	cp -r $(CUDA_DIR)/samples/0_Simple/matrixMul $(PWD)
	make -C matrixMul clean

matrixMul/matrixMul : matrixMul
	make -C matrixMul \
		NVCCFLAGS="-m64 -cudart shared" \
		GENCODE_FLAGS="-arch=$(ARCH)" \
		CPATH="$(CUDA_DIR)/samples/common/inc"

bandwidthTest :
	mkdir -p $(PWD)/bandwidthTest
	cp -r $(CUDA_DIR)/samples/1_Utilities/bandwidthTest $(PWD)
	make -C bandwidthTest clean

bandwidthTest/bandwidthTest : bandwidthTest
	make -C bandwidthTest \
		NVCCFLAGS="-m64 -cudart shared" \
		GENCODE_FLAGS="-arch=$(ARCH)" \
		CPATH="$(CUDA_DIR)/samples/common/inc"


clean :
	rm -f *.elf *.hex *.o *.d .depend *~
	make -C matrixMul clean
	make -C bandwidthTest clean

distclean : clean
	rm -r matrixMul
	rm -r bandwidthTest

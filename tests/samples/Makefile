CC = gcc
LD = gcc
CFLAGS = -Wall -std=gnu99
ARCH = sm_61
CUDA_DIR = /usr/local/cuda

PWD = $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

.PHONY: all depend clean dmtcpclean

all : matrixMul/matrixMul

matrixMul :
	mkdir -p $(PWD)/matrixMul
	cp -r $(CUDA_DIR)/samples/0_Simple/matrixMul $(PWD)
	make -C matrixMul clean

matrixMul/matrixMul : matrixMul
	make -C matrixMul \
		NVCCFLAGS="-m64 -cudart shared" \
		GENCODE_FLAGS="-arch=$(ARCH)" \
		CPATH="$(CUDA_DIR)/samples/common/inc"

clean :
	rm -f *.elf *.hex *.o *.d .depend *~
	make -C matrixMul clean

distclean : clean
	rm -r matrixMul

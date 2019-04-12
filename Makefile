#MIT License...

GDB_SRC = /home/nei/projects/cuda-gdb
CUDA_SRC = /usr/local/cuda
CC = gcc
MAKE = make

.PHONY: all cricket tests
all : cricket test

cricket :
	$(MAKE) -C src GDB_SRC=$(GDB_SRC) CUDA_SRC=$(CUDA_SRC) CC=$(CC)

tests :
	$(MAKE) -C tests CC=$(CC)

clean :
	rm -f *.elf *.hex *.o *.d .depend *~

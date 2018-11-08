#License...

GDB_SRC = /home/nei/projects/cuda-gdb
CUDA_SRC = /usr/local/cuda

CC = gcc
INCLUDES = -Iinclude/bfd -Iinclude/gdb -Iinclude/include -Iinclude/gdb/common
#INCLUDES = -I$(GDB_SRC)/build/bfd -I$(GDB_SRC)/build/gdb -I$(GDB_SRC)/include/ -I$(GDB_SRC)/gdb -I$(GDB_SRC)/gdb/common/ -I$(GDB_SRC)/bfd/ -I$(CUDA_SRC)/include
DLIBS = -lncurses -lpthread -lm -lz -ldl -Wl,--dynamic-list=$(GDB_SRC)/gdb/proc-service.list
SLIBS = libgdb.a libbfd.a libiberty.a libreadline.a libdecnumber.a libcudacore.a libopcodes.a
CFLAGS = -std=gnu99 $(INCLUDES)
LD = gcc
LDFLAGS = $(CFLAGS) $(DLIBS)
BINARY = cricket

FILES := debug.o \
		 cricket-cr.o \
		 cricket-device.o \
		 cricket-stack.o \
		 cricket-register.o \
		 cricket-file.o \
		 cricket-heap.o \
		 cricket-elf.o

.PHONY: all depend clean
all : $(BINARY)

%.o : %.c
	$(CC) $(CFLAGS) -c -o $@ $<

$(BINARY) : $(FILES)
	$(LD) $(LDFLAGS) -o $@ $^ $(SLIBS)

#.depend : $(FILES:.o=.cu)
#	$(CC) -MM $(CFLAGS) $(FILES:.o=.cu) > $@

#depend : .depend

clean :
	rm -f *.elf *.hex *.o *.d .depend *~ $(BINARY)

#include .depend

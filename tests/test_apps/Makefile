#License...

HOST_CC = gcc
HOST_LD = gcc
HOST_CFLAGS = -Wall -std=gnu99

CC = nvcc -ccbin g++
ARCH = sm_61
CFLAGS = -arch=$(ARCH) -cudart shared
#CFLAGS = -arch=$(ARCH)
LD = nvcc -ccbin g++
LDFLAGS = -arch=$(ARCH) -cudart shared
#LDFLAGS = -lcuda -arch=$(ARCH)
TEST_CPU_BIN = cpu.testapp
TEST_CPU_O = test_cpu.o
TEST_API_BIN = api.testapp
TEST_API_O = test_api.o
TEST_KERNEL_BIN = kernel.testapp
TEST_KERNEL_O = test_kernel.o
BINARY = cricket.testapp

LIBCUDA_WRAPPER = libcuda.so.1
LIBCUDA_OBJ = libcuda.o
LIBCUDA_LIBS = -ldl

FILES := matmul.cu

.PHONY: all depend clean 
all : $(TEST_KERNEL_BIN)

$(TEST_CPU_O) : $(FILES)
	$(CC) -DTEST_CPU $(CFLAGS) -dc -o $@ $<

$(TEST_API_O) : $(FILES)
	$(CC) -DTEST_API $(CFLAGS) -dc -o $@ $<

$(TEST_KERNEL_O) : $(FILES)
	$(CC) -DTEST_KERNEL $(CFLAGS) -dc -o $@ $<

%.o : %.cu
	$(CC) $(CFLAGS) -dc -o $@ $<

$(TEST_CPU_BIN) : $(TEST_CPU_O)
	$(LD) $(LDFLAGS) -o $@ $^

$(TEST_API_BIN) : $(TEST_API_O)
	$(LD) $(LDFLAGS) -o $@ $^

$(TEST_KERNEL_BIN) : $(TEST_KERNEL_O)
	$(LD) $(LDFLAGS) -o $@ $^

$(BINARY) : $(FILES)
	$(LD) $(LDFLAGS) -o $@ $^

$(LIBCUDA_OBJ) : $(LIBCUDA_OBJ:.o=.c)
	$(HOST_CC) -c -fpic -o $@ $< $(LIBCUDA_LIBS)

$(LIBCUDA_WRAPPER) : $(LIBCUDA_OBJ)
	$(HOST_LD) -shared -o $@ $^


clean :
	rm -f *.elf *.hex *.o *.d .depend *~ $(BINARY) $(LIBCUDA_WRAPPER) $(TEST_CPU_BIN) $(TEST_API_BIN) $(TEST_KERNEL_BIN)



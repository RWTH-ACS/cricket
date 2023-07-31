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
DEBUG_FLAGS = #-g -G
#LDFLAGS = -lcuda -arch=$(ARCH)
TEST_CPU_BIN = cpu.testapp
TEST_CPU_O = test_cpu.o
TEST_API_BIN = api.testapp
TEST_API_O = test_api.o
TEST_KERNEL_BIN = kernel.testapp
TEST_KERNEL_O = test_kernel.o
BINARY = cricket.testapp
OVERHEAD_BIN = overhead.testapp
OVERHEAD_O = overhead.o

TEST_KERNEL_LIB_O = test_kernel_lib.o
TEST_KERNEL_LIB = test_kernel.so
TEST_KERNEL_LIB_CALL_O = test_kernel_call.o
TEST_KERNEL_LIB_CALL = test_kernel_call

LIBCUDA_WRAPPER = libcuda.so.1
LIBCUDA_OBJ = libcuda.o
LIBCUDA_LIBS = -ldl -I../../cpu

FILES := matmul.cu

.PHONY: all depend clean 
all : $(TEST_KERNEL_BIN) $(BINARY) $(TEST_CPU_BIN) $(TEST_API_BIN) $(TEST_KERNEL_LIB) $(TEST_KERNEL_LIB_CALL)

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

$(OVERHEAD_O) : $(OVERHEAD_O:.o=.cu)
	$(CC) -O3 $(CFLAGS) -dc -o $@ $<

$(OVERHEAD_BIN) : $(OVERHEAD_O)
	$(LD) $(LDFLAGS) -O2 -o $@ $^

$(LIBCUDA_OBJ) : $(LIBCUDA_OBJ:.o=.c)
	$(HOST_CC) -c -fpic -o $@ $< $(LIBCUDA_LIBS)

$(TEST_KERNEL_LIB_O) : $(FILES)
	$(CC) $(CFLAGS) $(DEBUG_FLAGS) -dc --compiler-options '-fPIC' -o $@ $<

$(TEST_KERNEL_LIB) : $(TEST_KERNEL_LIB_O)
	$(LD) $(LDFLAGS) $(DEBUG_FLAGS) -shared -o lib$@ $^

$(TEST_KERNEL_LIB_CALL_O) : $(TEST_KERNEL_LIB_CALL_O:.o=.c)
	$(HOST_CC) -c -o $@ $<

$(TEST_KERNEL_LIB_CALL) : $(TEST_KERNEL_LIB_CALL_O)
	$(HOST_LD) -o $@ $< -I. -ldl

$(LIBCUDA_WRAPPER) : $(LIBCUDA_OBJ)
	$(HOST_LD) -shared -o $@ $^


clean :
	rm -f *.elf *.hex *.o *.d .depend *~ $(BINARY) $(LIBCUDA_WRAPPER) $(TEST_CPU_BIN) $(TEST_API_BIN) $(TEST_KERNEL_BIN) $(TEST_KERNEL_LIB) $(TEST_KERNEL_LIB_CALL) 



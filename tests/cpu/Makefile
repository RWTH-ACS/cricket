CUDA_SRC = /usr/local/cuda
LIBTIRPC_PREFIX = ../../submodules/libtirpc/install

CC = gcc
LD = gcc

SRC_SERVER_TESTS = test_list.c test_resource_mg.c

OBJ_SERVER_TESTS = $(SRC_SERVER_TESTS:%.c=%.o)
BIN_SERVER_TESTS = $(SRC_SERVER_TESTS:%.c=%.test)

# Compiler flags
INC_FLAGS += -I$(LIBTIRPC_PREFIX)/include/tirpc
INC_FLAGS += -I$(CUDA_SRC)/include
INC_FLAGS += -I../../cpu/

LIB_FLAGS += -L$(LIBTIRPC_PREFIX)/lib
LIB_FLAGS += -L$(CUDA_SRC)/lib64
LIB_FLAGS += -L../../cpu/
CC_FLAGS += -std=gnu99 $(INC_FLAGS) -g -ggdb -fsanitize=address -fsanitize=pointer-compare -fsanitize=pointer-subtract -fsanitize-address-use-after-scope
LD_FLAGS = $(LIB_FLAGS)

ifdef LOG
CC_FLAGS += -DLOG_LEVEL=LOG_$(LOG)
endif

SERVER_LD_FLAGS = $(LD_FLAGS) -l:cricket-server.so -lcuda
CLIENT_LD_FLAGS = $(LD_FLAGS) -l:cricket-client.so

# Targets
.PHONY: all clean

all : $(BIN_CLIENT_TESTS) $(BIN_SERVER_TESTS)

$(BIN_SERVER_TESTS) : %.test:%.o
	$(LD) $(CC_FLAGS) -o $@ $< $(SERVER_LD_FLAGS)

$(OBJ_SERVER_TESTS) : %.o:%.c
	$(CC) $(CC_FLAGS) -c -o $@ $< $(SERVER_LD_FLAGS)

clean:
	 rm -f $(OBJ_SERVER_TESTS) $(OBJ_CLIENT_TESTS) $(BIN_SERVER_TESTS) $(BIN_CLIENT_TESTS)

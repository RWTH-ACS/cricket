CLIENT = test-client
SERVER = cricket-server.so
LIBCUDA_WRAPPER = cricket-client.so

CUDA_SRC = /usr/local/cuda
LIBTIRPC_PREFIX = /home/eiling/projects/libtirpc/install


CC = gcc
LD = gcc


RPCGEN = rpcgen

RPC_SERVER = cpu_rpc_prot_svc.c
RPC_CLIENT = cpu_rpc_prot_clnt.c
RPC_XDR = cpu_rpc_prot_xdr.c
RPC_DEF = cpu_rpc_prot.x
RPC_H = $(RPC_DEF:%.x=%.h)

SRC_CLIENT = $(RPC_XDR) $(RPC_CLIENT) test-client.c
SRC_SERVER = $(RPC_XDR) $(RPC_SERVER) cpu-server.c cpu-utils.c cpu-server-runtime.c cpu-server-driver.c cpu-server-driver-hidden.c
SRC_LIBCUDA = $(RPC_XDR) $(RPC_CLIENT) cpu-client.c cpu-utils.c cpu-client-runtime.c cpu-client-driver.c cpu-client-driver-hidden.c

OBJ_CLIENT = $(SRC_CLIENT:%.c=%.o)
OBJ_SERVER = $(SRC_SERVER:%.c=%.o)
OBJ_LIBCUDA = $(SRC_LIBCUDA:%.c=%.o)

# Compiler flags

LIB_DIR = ../lib
# Order of .a files is important!
SLIBS = libgdb.a libiberty.a libreadline.a libdecnumber.a libcudacore.a libopcodes.a
SLIBS:= $(addprefix $(LIB_DIR)/, $(SLIBS))

INC_FLAGS += -I$(LIBTIRPC_PREFIX)/usr/include/tirpc
INC_FLAGS += -I$(CUDA_SRC)/include
#INC_FLAGS += -I../include/bfd -I../include/gdb -I../include/include -I../include/gdb/common -I../include

LIB_FLAGS += -L$(LIBTIRPC_PREFIX)/usr/lib -L$(CUDA_SRC)/lib64
CC_FLAGS += -std=gnu99 $(INC_FLAGS)
LD_FLAGS += $(LIB_FLAGS) -ltirpc -lcuda -lcudart
#SERVER_LD_FLAGS = $(LD_FLAGS) -lbfd -lncurses -lpthread -lm -lz -ldl -Wl,--dynamic-list=../utils/proc-service.list
SERVER_CC_FLAGS = -std=gnu99 -I$(LIBTIRPC_PREFIX)/usr/include/tirpc -I$(CUDA_SRC)/include -L$(LIBTIRPC_PREFIX)/usr/lib
SERVER_LD_FLAGS = $(LIB_FLAGS) -ltirpc -lcudart -lbfd -ldl

LD_LIBCUDA = $(LIB_FLAGS) -ltirpc -ldl

RPCGEN_FLAGS = -C -M -N

# Targets
.PHONY: all clean

all : $(SERVER) $(LIBCUDA_WRAPPER)

$(CLIENT) : $(OBJ_CLIENT)
	$(LD) $(CC_FLAGS) -o $@ $^ $(LD_FLAGS)

$(LIBCUDA_WRAPPER) : $(OBJ_LIBCUDA)
	$(LD) $(CC_FLAGS) -shared -o $@ $^ $(LD_LIBCUDA)

$(SERVER) : $(OBJ_SERVER)
	$(LD) $(SERVER_CC_FLAGS) -shared -o $@ $^ $(SERVER_LD_FLAGS)

$(RPC_H) : $(RPC_DEF)
	$(RPCGEN) $(RPCGEN_FLAGS) -h -o $@ $<

$(RPC_CLIENT) : $(RPC_DEF)
	$(RPCGEN) $(RPCGEN_FLAGS) -l -o $@ $<

$(RPC_SERVER) : $(RPC_DEF)
	$(RPCGEN) $(RPCGEN_FLAGS) -m -o $@ $<

$(RPC_XDR) : $(RPC_DEF)
	$(RPCGEN) $(RPCGEN_FLAGS) -c -o $@ $<

cricketd.o : cricketd.c $(RPC_H)
	$(CC) $(SERVER_CC_FLAGS) -c -fpic -o $@ $< $(SERVER_LD_FLAGS)

%.o : %.c $(RPC_H)
	$(CC) $(CC_FLAGS) -c -fpic -o $@ $< $(LD_FLAGS)

clean:
	 rm -f $(RPC_H) $(RPC_CLIENT) $(RPC_SERVER) $(RPC_XDR) $(OBJ_CLIENT) $(OBJ_SERVER) $(OBJ_LIBCUDA) $(CLIENT) $(SERVER) $(LIBCUDA_WRAPPER)




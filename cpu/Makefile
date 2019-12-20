CLIENT = client
SERVER = cricketd
LIBCUDA_WRAPPER = libcudawrap.so

CUDA_SRC = /usr/local/cuda
LIBTIRPC_PREFIX = /home/eiling/projects/libtirpc/install


CC = gcc
LD = gcc


RPCGEN = rpcgen

RPC_SERVER = rpc_prot_svc.c
RPC_CLIENT = rpc_prot_clnt.c
RPC_XDR = rpc_prot_xdr.c
RPC_DEF = cd_rpc_prot.x
RPC_H = $(RPC_DEF:%.x=%.h)

SRC_CLIENT = $(RPC_XDR) $(RPC_CLIENT) client.c
SRC_SERVER = $(RPC_XDR) $(RPC_SERVER) cricketd.c
SRC_LIBCUDA = $(RPC_XDR) $(RPC_CLIENT) cd_libcuda_api.c

OBJ_CLIENT = $(SRC_CLIENT:%.c=%.o)
OBJ_SERVER = $(SRC_SERVER:%.c=%.o)
OBJ_LIBCUDA = $(SRC_LIBCUDA:%.c=%.o)

# Compiler flags

INC_FLAGS += -I$(LIBTIRPC_PREFIX)/usr/include/tirpc
INC_FLAGS += -I$(CUDA_SRC)/include

LIB_FLAGS += -L$(LIBTIRPC_PREFIX)/usr/lib
CC_FLAGS += -std=gnu99 $(INC_FLAGS)
LD_FLAGS += $(LIB_FLAGS) -ltirpc -lcuda -lcudart

LD_LIBCUDA = $(LIB_FLAGS) -ltirpc -ldl

RPCGEN_FLAGS = -C -M -N

# Targets
.PHONY: all clean

all : $(CLIENT) $(SERVER)

$(CLIENT) : $(OBJ_CLIENT)
	$(LD) $(CC_FLAGS) -o $@ $^ $(LD_FLAGS)

$(LIBCUDA_WRAPPER) : $(OBJ_LIBCUDA)
	$(LD) $(CC_FLAGS) -shared -o $@ $^ $(LD_LIBCUDA)

$(SERVER) : $(OBJ_SERVER)
	$(LD) $(CC_FLAGS) -o $@ $^ $(LD_FLAGS)

$(RPC_H) : $(RPC_DEF)
	$(RPCGEN) $(RPCGEN_FLAGS) -h -o $@ $<

$(RPC_CLIENT) : $(RPC_DEF)
	$(RPCGEN) $(RPCGEN_FLAGS) -l -o $@ $<

$(RPC_SERVER) : $(RPC_DEF)
	$(RPCGEN) $(RPCGEN_FLAGS) -m -o $@ $<

$(RPC_XDR) : $(RPC_DEF)
	$(RPCGEN) $(RPCGEN_FLAGS) -c -o $@ $<

%.o : %.c $(RPC_H)
	$(CC) $(CC_FLAGS) -c -fpic -o $@ $< $(LD_FLAGS)

clean:
	 rm -f $(RPC_H) $(RPC_CLIENT) $(RPC_SERVER) $(RPC_XDR) $(OBJ_CLIENT) $(OBJ_SERVER) $(OBJ_LIBCUDA) $(CLIENT) $(SERVER) $(LIBCUDA_WRAPPER)




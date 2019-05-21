CLIENT = client
SERVER = cricketd

CUDA_SRC = /usr/local/cuda

CC = gcc
LD = gcc


RPCGEN = rpcgen

RPC_SERVER = rpc_prot_svc.c
RPC_CLIENT = rpc_prot_clnt.c
RPC_XDR = rpc_prot_xdr.c
RPC_DEF = cd_rpc_prot.x
RPC_H = $(RPC_DEF:%.x=%.h)

SRC_CLIENT = $(RPC_CLIENT) $(RPC_XDR) client.c
SRC_SERVER = $(RPC_SERVER) $(RPC_XDR) cricketd.c
OBJ_CLIENT = $(SRC_CLIENT:%.c=%.o)
OBJ_SERVER = $(SRC_SERVER:%.c=%.o)
# Compiler flags 

INC_FLAGS += -I/usr/include/tirpc
CC_FLAGS += -std=gnu99
LD_FLAGS += -ltirpc 
RPCGEN_FLAGS = -C -M -N

# Targets 
.PHONY: all clean

all : $(CLIENT) $(SERVER)

$(CLIENT) : $(OBJ_CLIENT)
	$(LD) $(CC_FLAGS) -o $@ $^ $(LD_FLAGS)

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
	$(CC) $(CC_FLAGS) $(INC_FLAGS) -c -o $@ $< $(LD_FLAGS)
	
clean:
	 rm -f $(RPC_H) $(RPC_CLIENT) $(RPC_SERVER) $(RPC_XDR) $(OBJ_CLIENT) $(OBJ_SERVER) $(CLIENT) $(SERVER)



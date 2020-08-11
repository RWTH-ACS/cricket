#RPC server library
SERVER = cricket-server.so
#RPC client library
CLIENT = cricket-client.so

CUDA_SRC = /usr/local/cuda
LIBTIRPC_PREFIX = ../submodules/libtirpc/install

CC = gcc
LD = gcc
RPCGEN = rpcgen

# rpcgen files
RPC_SERVER = cpu_rpc_prot_svc.c
RPC_CLIENT = cpu_rpc_prot_clnt.c
RPC_XDR = cpu_rpc_prot_xdr.c
RPC_DEF = cpu_rpc_prot.x
RPC_H = $(RPC_DEF:%.x=%.h)


SRC_SERVER = $(RPC_XDR)                 \
			 $(RPC_SERVER)              \
			 cpu-server.c               \
			 cpu-utils.c                \
			 cpu-server-runtime.c       \
			 cpu-server-driver.c        \
			 cpu-server-driver-hidden.c \
			 log.c                      \
			 cpu-libwrap.c              \
			 cpu-server-cusolver.c

SRC_CLIENT = $(RPC_XDR)                 \
			 $(RPC_CLIENT)              \
			 cpu-client.c               \
			 cpu-utils.c                \
			 cpu-client-runtime.c       \
			 cpu-client-driver.c        \
			 cpu-client-driver-hidden.c \
			 log.c                      \
			 cpu-libwrap.c              \
			 cpu-client-cusolver.c

ifdef WITH_IB
SRC_SERVER += cpu-ib.c
SRC_CLIENT += cpu-ib.c
endif

OBJ_SERVER = $(SRC_SERVER:%.c=%.o)
OBJ_CLIENT = $(SRC_CLIENT:%.c=%.o)

# Compiler flags
RPCGEN_FLAGS = -C -M -N
INC_FLAGS += -I$(LIBTIRPC_PREFIX)/include/tirpc
INC_FLAGS += -I$(CUDA_SRC)/include

LIB_FLAGS += -L$(LIBTIRPC_PREFIX)/lib -L$(CUDA_SRC)/lib64
CC_FLAGS += -std=gnu99 $(INC_FLAGS)
LD_FLAGS = $(LIB_FLAGS) -ltirpc -ldl -lcrypto

ifdef WITH_IB
LD_FLAGS += -libverbs
endif

ifdef LOG
CC_FLAGS += -DLOG_LEVEL=LOG_$(LOG)
endif
ifdef WITH_IB
CC_FLAGS += -DWITH_IB=$(WITH_IB)
endif

SERVER_LD_FLAGS = $(LD_FLAGS) -lcudart -lcusolver -lbfd
CLIENT_LD_FLAGS = $(LD_FLAGS)

# Targets
.PHONY: all clean

all : $(SERVER) $(CLIENT)

$(CLIENT) : $(OBJ_CLIENT)
	$(LD) $(CC_FLAGS) -shared -o $@ $^ $(CLIENT_LD_FLAGS)

$(SERVER) : $(OBJ_SERVER)
	$(LD) $(CC_FLAGS) -shared -o $@ $^ $(SERVER_LD_FLAGS)

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
	 rm -f $(RPC_H) $(RPC_CLIENT) $(RPC_SERVER) $(RPC_XDR) $(OBJ_CLIENT) $(OBJ_SERVER) $(SERVER) $(CLIENT)




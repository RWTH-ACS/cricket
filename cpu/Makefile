#Standalone RPC Server
SERVER = cricket-rpc-server
SERVER_LIB = cricket-server.so
#RPC client library
CLIENT = cricket-client.so

CUDA_SRC = /usr/local/cuda
LIBTIRPC_PREFIX = ../submodules/libtirpc/install
SUBMODULE_LIBS = ../submodules/lib

CC = gcc
LD = gcc
RPCGEN = rpcgen

# rpcgen files
RPC_SERVER = cpu_rpc_prot_svc.c
RPC_SERVER_MOD = cpu_rpc_prot_svc_mod.c
RPC_CLIENT = cpu_rpc_prot_clnt.c
RPC_XDR = cpu_rpc_prot_xdr.c
RPC_DEF = cpu_rpc_prot.x
RPC_H = $(RPC_DEF:%.x=%.h)



SRC_SERVER = $(RPC_XDR)                 \
			 $(RPC_SERVER_MOD)          \
			 cpu-server.c               \
			 cpu-utils.c                \
			 cpu-server-runtime.c       \
			 cpu-server-driver.c        \
			 cpu-server-driver-hidden.c \
			 log.c                      \
			 cpu-libwrap.c              \
			 cpu-server-cusolver.c 		\
			 cpu-server-cublas.c 		\
			 list.c                     \
			 api-recorder.c 			\
			 resource-mg.c 			    \
			 cr.c 					    \
			 gsched_none.c 			    \
			 oob.c 					    \
			 mt-memcpy.c				\
			 cpu-elf2.c

SRC_SERVER_LIB = server-library.c
SRC_SERVER_EXE = server-exe.c

SRC_CLIENT = $(RPC_XDR)                 \
			 $(RPC_CLIENT)              \
			 cpu-client.c               \
			 cpu-utils.c                \
			 cpu-client-runtime.c       \
			 cpu-client-driver.c        \
			 list.c 				    \
			 log.c                      \
			 cpu-libwrap.c              \
			 cpu-client-cusolver.c 		\
			 oob.c 					    \
			 mt-memcpy.c				\
			 cpu-elf2.c

# 			 cpu-client-driver-hidden.c \

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

LIB_FLAGS += -L$(LIBTIRPC_PREFIX)/lib
LIB_FLAGS += -L$(CUDA_SRC)/lib64
CC_FLAGS += -std=gnu99 $(INC_FLAGS) -O2
# TODO: use extern in header files instead of direct definition e.g. in cpu-common.h to remove -fcommon flag
CC_FLAGS += -fcommon
LD_FLAGS = $(LIB_FLAGS) -ltirpc -ldl -lcrypto -lelf

ifdef WITH_DEBUG
# use ASAN_OPTIONS=protect_shadow_gap=0  LSAN_OPTIONS=fast_unwind_on_malloc=0 when running
CC_FLAGS += -g -ggdb #-fsanitize=address -fsanitize=pointer-compare -fsanitize=pointer-subtract -fsanitize-address-use-after-scope
endif

ifdef WITH_IB
LD_FLAGS += -libverbs
endif

ifdef LOG
CC_FLAGS += -DLOG_LEVEL=LOG_$(LOG)
endif

ifdef LOGN
CC_FLAGS += -DLOG_LEVEL=$(LOGN)
endif

ifdef WITH_IB
CC_FLAGS += -DWITH_IB=$(WITH_IB)
endif

SERVER_LD_FLAGS = $(LD_FLAGS) -lcudart -lcusolver -lcuda -lcublas -lrt -lpthread
SERVER_BIN_LD_FLAGS = $(SERVER_LD_FLAGS) -Wl,--unresolved-symbols=ignore-in-object-files
CLIENT_LD_FLAGS = $(LD_FLAGS)

# Targets
.PHONY: all clean

all : $(SERVER) $(CLIENT)

$(CLIENT) : $(OBJ_CLIENT)
	$(LD) $(CC_FLAGS) -shared -o $@ $^ $(CLIENT_LD_FLAGS)

$(SERVER_LIB) : $(OBJ_SERVER) $(SRC_SERVER_EXE:%.c=%.o)
	$(LD) $(CC_FLAGS) -shared -o $@ $^ $(SERVER_BIN_LD_FLAGS)

$(SERVER) : $(OBJ_SERVER) $(SRC_SERVER_EXE:%.c=%.o)
	$(LD) $(CC_FLAGS) -o $@ $^ $(SERVER_BIN_LD_FLAGS)

$(RPC_H) : $(RPC_DEF)
	rm -f $@ && $(RPCGEN) $(RPCGEN_FLAGS) -h -o $@ $<

$(RPC_CLIENT) : $(RPC_DEF)
	rm -f $@ && $(RPCGEN) $(RPCGEN_FLAGS) -l -o $@ $<

$(RPC_SERVER) : $(RPC_DEF)
	rm -f $@ && $(RPCGEN) $(RPCGEN_FLAGS) -m -o $@ $<

$(RPC_SERVER_MOD) : $(RPC_SERVER)
	./generate_dispatch.sh

$(RPC_XDR) : $(RPC_DEF)
	rm -f $@ && $(RPCGEN) $(RPCGEN_FLAGS) -c -o $@ $<

%.o : %.c $(RPC_H)
	$(CC) $(CC_FLAGS) -c -fpic -o $@ $< $(LD_FLAGS) 

clean:
	 rm -f $(RPC_H) $(RPC_CLIENT) $(RPC_SERVER) $(RPC_SERVER_MOD) $(RPC_XDR) $(OBJ_CLIENT) $(OBJ_SERVER) $(SERVER) $(SERVER_LIB) $(CLIENT) $(SRC_SERVER_EXE:%.c=%.o)





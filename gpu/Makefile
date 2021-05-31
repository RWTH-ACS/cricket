#MIT License...
BINARY = cricket

CUDA_SRC = /usr/local/cuda
CUDA_GDB_PATH = ../submodules/cuda-gdb

CC = g++
LD = g++

SRC_DIR := src
INC_DIRS := -I$(CUDA_GDB_PATH)/bfd \
			-I$(CUDA_GDB_PATH)/gdb \
			-I$(CUDA_GDB_PATH)/include \
			-I$(CUDA_GDB_PATH)/gdb/common \
			-I$(CUDA_GDB_PATH)/build/gdb \
			-I$(CUDA_GDB_PATH)/gdb/gnulib/import \
			-I$(CUDA_GDB_PATH)/build/bfd \
			-I$(CUDA_GDB_PATH)/build/gdb/build-gnulib/import \
			-Iinclude
LIB_DIR := ../submodules/lib
BUILD_DIR := build

DLIBS = -lncurses -lpthread -lm -lz -ldl -lexpat -llzma -Wl,--dynamic-list=utils/proc-service.list
# Order of .a files is important!
SLIBS = libgdb.a libbfd.a libiberty.a libreadline.a libdecnumber.a libcudacore.a libopcodes.a libgnu.a
SLIBS:= $(addprefix $(LIB_DIR)/, $(SLIBS))
CFLAGS = -std=gnu++11 -fpermissive -g#-fsanitize=address -fsanitize=pointer-compare -fsanitize=pointer-subtract -fsanitize-address-use-after-scope
LDFLAGS = $(CFLAGS)
# generate the names like src/main.o src/main.d
SRCS_C := $(wildcard $(SRC_DIR)/*.c)
SRCS_BASE := $(patsubst %.c,%,$(SRCS_C))
BUILD_BASE := $(addprefix $(BUILD_DIR)/, $(notdir $(SRCS_BASE)))
DEPS := $(addsuffix .d,$(BUILD_BASE))
OBJS := $(addsuffix .o,$(BUILD_BASE))

ifdef LOG
CFLAGS += -DLOG_LEVEL=LOG_$(LOG)
endif


.PHONY: all objs clean show

all: $(DEPS) $(BINARY)

objs: $(OBJS)

build:  # create the build/ directory if it doesn't exist
	mkdir -p $@

$(BINARY): $(OBJS) | build
	@echo -e "\033[34m----> Linking $@ \033[0m"
	$(LD) $(LDFLAGS) $(DLIBS) -o $@ $^ $(SLIBS)

$(BUILD_DIR)/%.d: $(SRC_DIR)/%.c | build
	@echo -e "\033[32m----> Generate dependency file for $<\033[0m"
	$(CC) $(CFLAGS) $(INC_DIRS) -MM -MT$(@:.d=.o) -o $@ $<

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | build
	@echo -e "\033[33m----> Compiling $<\033[0m"
	$(CC) $(CFLAGS) $(INC_DIRS) -c $< -o $@

clean:
	@echo -e "\033[31m----> Cleaning up\033[0m"
	rm -rf $(BUILD_DIR) $(BINARY)

show: # For debugging purposes
	@echo -e '\033[36mBINARY      \033[0m' $(BINARY)
	@echo -e '\033[36mSRCS_C      \033[0m' $(SRCS_C)
	@echo -e '\033[36mDLIBS       \033[0m' $(DLIBS)
	@echo -e '\033[36mSLIBS       \033[0m' $(SLIBS)
	@echo -e '\033[36mSOURCE_BASE \033[0m' $(SRCS_BASE)
	@echo -e '\033[36mBUILD_BASE  \033[0m' $(BUILD_BASE)
	@echo -e '\033[36mDEPS        \033[0m' $(DEPS)
	@echo -e '\033[36mOBJS        \033[0m' $(OBJS)

#include $(wildcard build/*.d)

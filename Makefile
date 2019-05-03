#MIT License...
BINARY = cricket

CUDA_SRC = /usr/local/cuda

CC = gcc
LD = gcc

SRC_DIR := src
INC_DIRS := -Iinclude/bfd -Iinclude/gdb -Iinclude/include -Iinclude/gdb/common -Iinclude
LIB_DIR := lib
BUILD_DIR := build

DLIBS = -lncurses -lpthread -lm -lz -ldl -Wl,--dynamic-list=utils/proc-service.list
# Order of .a files is important!
SLIBS = libgdb.a libbfd.a libiberty.a libreadline.a libdecnumber.a libcudacore.a libopcodes.a
SLIBS:= $(addprefix $(LIB_DIR)/, $(SLIBS))
CFLAGS = -std=gnu99
LDFLAGS = $(CFLAGS)
# generate the names like src/main.o src/main.d
SRCS_C := $(wildcard $(SRC_DIR)/*.c)
SRCS_BASE := $(patsubst %.c,%,$(SRCS_C))
BUILD_BASE := $(addprefix $(BUILD_DIR)/, $(notdir $(SRCS_BASE)))
DEPS := $(addsuffix .d,$(BUILD_BASE))
OBJS := $(addsuffix .o,$(BUILD_BASE))


.PHONY: all objs clean show tests format

all: $(DEPS) $(BINARY) tests

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
	@echo -e "\033[31m----> Cleaning up test kernels\033[0m"
	$(MAKE) -C tests clean

tests:
	@echo -e "\033[36m----> Building test kernels\033[0m"
	$(MAKE) -C tests

format:
	@echo -e "\033[35m----> Formatting source code\033[0m"
	echo "Formatting C files"
	clang-format -i ${SRCS_C}
	echo "Formatting Header files"
	clang-format -i include/*.h

show: # For debugging purposes
	@echo -e '\033[36mBINARY      \033[0m' $(BINARY)
	@echo -e '\033[36mSRCS_C      \033[0m' $(SRCS_C)
	@echo -e '\033[36mDLIBS       \033[0m' $(DLIBS)
	@echo -e '\033[36mSLIBS       \033[0m' $(SLIBS)
	@echo -e '\033[36mSOURCE_BASE \033[0m' $(SRCS_BASE)
	@echo -e '\033[36mBUILD_BASE  \033[0m' $(BUILD_BASE)
	@echo -e '\033[36mDEPS        \033[0m' $(DEPS)
	@echo -e '\033[36mOBJS        \033[0m' $(OBJS)

include $(wildcard build/*.d)

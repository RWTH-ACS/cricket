#MIT License...
SONAME = libchimp.so
P_ROOT = ../..

CC = gcc
LD = gcc

SRC_DIR := .
INC_DIRS := -I$(P_ROOT)/include/chimp
LIB_DIR := $(P_ROOT)/lib
BUILD_DIR := $(P_ROOT)/build/chimp

DLIBS = -ldl
# Order of .a files is important!
SLIBS = 
SLIBS:= $(addprefix $(LIB_DIR)/, $(SLIBS))
CFLAGS = -std=gnu99 -fpic
LDFLAGS = -shared $(CFLAGS)
# generate the names like src/main.o src/main.d
SRCS_C := $(wildcard $(SRC_DIR)/*.c)
SRCS_BASE := $(patsubst %.c,%,$(SRCS_C))
BUILD_BASE := $(addprefix $(BUILD_DIR)/, $(notdir $(SRCS_BASE)))
DEPS := $(addsuffix .d,$(BUILD_BASE))
OBJS := $(addsuffix .o,$(BUILD_BASE))


.PHONY: all objs clean show

all: $(DEPS) $(LIB_DIR)/$(SONAME)

objs: $(OBJS)

$(BUILD_DIR):  # create the build/ directory if it doesn't exist
	mkdir -p $@

$(SONAME): $(OBJS) | $(BUILD_DIR)
	@echo -e "\033[34m----> Linking $@ \033[0m"
	$(LD) $(LDFLAGS) $(DLIBS) -o $@ $^ $(SLIBS)

$(LIB_DIR)/$(SONAME): $(SONAME)
	cp $^ $@

$(BUILD_DIR)/%.d: $(SRC_DIR)/%.c | $(BUILD_DIR)
	@echo -e "\033[32m----> Generate dependency file for $<\033[0m"
	$(CC) $(CFLAGS) $(INC_DIRS) -MM -MT$(@:.d=.o) -o $@ $<

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	@echo -e "\033[33m----> Compiling $<\033[0m"
	$(CC) $(CFLAGS) $(INC_DIRS) -c $< -o $@

clean:
	@echo -e "\033[31m----> Cleaning up\033[0m"
	rm -rf $(BUILD_DIR) $(SONAME)
	@echo -e "\033[31m----> Cleaning up test kernels\033[0m"

show: # For debugging purposes
	@echo -e '\033[36mSONAME      \033[0m' $(SONAME)
	@echo -e '\033[36mSRCS_C      \033[0m' $(SRCS_C)
	@echo -e '\033[36mDLIBS       \033[0m' $(DLIBS)
	@echo -e '\033[36mSLIBS       \033[0m' $(SLIBS)
	@echo -e '\033[36mSOURCE_BASE \033[0m' $(SRCS_BASE)
	@echo -e '\033[36mBUILD_BASE  \033[0m' $(BUILD_BASE)
	@echo -e '\033[36mDEPS        \033[0m' $(DEPS)
	@echo -e '\033[36mOBJS        \033[0m' $(OBJS)

include $(wildcard $(BUILD_DIR)/*.d)

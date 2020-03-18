#MIT License...
.PHONY: all libtirpc clean

all: libtirpc

clean:
	@echo -e "\033[31m----> Cleaning up libtirpc\033[0m"
	$(MAKE) -C libtirpc clean

libtirpc:
	@echo -e "\033[36m----> autogen libtirpc\033[0m"
	if [ ! -f "libtirpc/configure" ]; then cd libtirpc && ./bootstrap; fi
	@echo -e "\033[36m----> Configuring libtirpc\033[0m"
	cd libtirpc && ./configure --disable-gssapi --prefix=$(dir $(realpath $(firstword $(MAKEFILE_LIST))))/libtirpc/install
	@echo -e "\033[36m----> Building libtirpc\033[0m"
	$(MAKE) -C libtirpc
	@echo -e "\033[36m----> Installing libtirpc to ./libtirpc/install\033[0m"
	$(MAKE) -C libtirpc install

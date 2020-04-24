#MIT License...
.PHONY: all libtirpc cuda-gdb clean

all: libtirpc cuda-gdb

clean:
	@echo -e "\033[31m----> Cleaning up libtirpc\033[0m"
	$(MAKE) -C libtirpc clean
	rm -rf cuda-gdb/build
	cd cuda-gdb && git apply -R ../cuda-gdb.patch
	rm -rf lib

libtirpc:
	@echo -e "\033[36m----> autogen libtirpc\033[0m"
	if [ ! -f "libtirpc/configure" ]; then cd libtirpc && ./bootstrap; fi
	@echo -e "\033[36m----> Configuring libtirpc\033[0m"
	cd libtirpc && ./configure --disable-gssapi --prefix=$(dir $(realpath $(firstword $(MAKEFILE_LIST))))/libtirpc/install
	@echo -e "\033[36m----> Building libtirpc\033[0m"
	$(MAKE) -C libtirpc
	@echo -e "\033[36m----> Installing libtirpc to ./libtirpc/install\033[0m"
	$(MAKE) -C libtirpc install

cuda-gdb/build:
	@echo -e "\033[36m----> patching cuda-gdb\033[0m"
	cd cuda-gdb && git apply ../cuda-gdb.patch
	@echo -e "\033[36m----> Configuring cuda-gdb\033[0m"
	mkdir -p cuda-gdb/build && cd cuda-gdb/build && \
		../configure --disable-werror --program-prefix=cuda- --enable-cuda --enable-targets="x86_64-apple-darwin,x86_64-unknown-linux-gnu,arm-elf-linux-gnu,m68k-unknown-linux-gnu" CFLAGS='-I/usr/local/cuda/include' LDFLAGS='-lpthread'
	@echo -e "\033[36m----> Building cuda-gdb\033[0m"
	$(MAKE) -C cuda-gdb/build
	$(MAKE) -C cuda-gdb/build/gdb libgdb.a

lib:
	mkdir -p lib

lib/libbfd.a: cuda-gdb/build lib
	cp $</bfd/libbfd.a $@

lib/libcudacore.a: cuda-gdb/build lib
	cp $</libcudacore/libcudacore.a $@

lib/libgdb.a: cuda-gdb/build lib
	cp $</gdb/libgdb.a $@

lib/libiberty.a: cuda-gdb/build lib
	cp $</libiberty/libiberty.a $@

lib/libopcodes.a: cuda-gdb/build lib
	cp $</opcodes/libopcodes.a $@

lib/libreadline.a: cuda-gdb/build lib
	cp $</readline/libreadline.a $@

lib/libdecnumber.a: cuda-gdb/build lib
	cp $</libdecnumber/libdecnumber.a $@

cuda-gdb: lib/libbfd.a lib/libcudacore.a lib/libgdb.a lib/libiberty.a lib/libopcodes.a lib/libreadline.a lib/libdecnumber.a
	@echo -e "\033[36m----> cuda-gdb installed to lib\033[0m"

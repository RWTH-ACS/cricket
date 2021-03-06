#MIT License...
.PHONY: all libtirpc cuda-gdb clean

all: libtirpc cuda-gdb-libs

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

cuda-gdb:
	@echo -e "\033[36m----> downloading cuda-gdb 11.1\033[0m"
	wget https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-gdb-src-11-1-11.1.105-1.x86_64.rpm
	@echo -e "\033[36m----> extracting cuda-gdb\033[0m"
	mkdir -p cuda-gdb
	rpm2cpio ./cuda-gdb-src-11-1-11.1.105-1.x86_64.rpm | cpio -i --to-stdout ./usr/local/cuda-11.1/extras/cuda-gdb-11.1.105.src.tar.gz | tar zxf - -C cuda-gdb --strip-components=1

cuda-gdb/build:
	@echo -e "\033[36m----> Configuring cuda-gdb\033[0m"
	mkdir -p cuda-gdb/build && cd cuda-gdb/build && \
		../configure --disable-werror --program-prefix=cuda- --enable-cuda --enable-targets="x86_64-apple-darwin,x86_64-unknown-linux-gnu,arm-elf-linux-gnu,m68k-unknown-linux-gnu" CFLAGS='-I/usr/local/cuda/include' LDFLAGS='-lpthread'
	@echo -e "\033[36m----> Building cuda-gdb\033[0m"
	CPATH=/usr/local/cuda/include $(MAKE) -C cuda-gdb/build
	CPATH=/usr/local/cuda/include $(MAKE) -C cuda-gdb/build/gdb libgdb.a

lib:
	mkdir -p lib

lib/libbfd.a: cuda-gdb/build lib
	cp $</bfd/libbfd.a $@

lib/libgnu.a: cuda-gdb/build lib
	cp $</gdb/build-gnulib/import/libgnu.a $@

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

cuda-gdb-libs: lib/libbfd.a lib/libcudacore.a lib/libgdb.a lib/libiberty.a lib/libopcodes.a lib/libreadline.a lib/libdecnumber.a lib/libgnu.a
	@echo -e "\033[36m----> cuda-gdb installed to lib\033[0m"

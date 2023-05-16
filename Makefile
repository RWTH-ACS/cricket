#MIT License...

.PHONY: all cuda-gdb libtirpc gpu cpu tests clean install install-cpu bin/tests

all: gpu cpu install

clean:
	@echo -e "\033[31m----> Cleaning up gpu\033[0m"
	$(MAKE) -C gpu clean
	@echo -e "\033[31m----> Cleaning up cpu\033[0m"
	$(MAKE) -C cpu clean
	@echo -e "\033[31m----> Cleaning up test kernels\033[0m"
	$(MAKE) -C tests clean

cuda-gdb:
	@echo -e "\033[36m----> Building submodules\033[0m"
	$(MAKE) -C submodules cuda-gdb
	$(MAKE) -C submodules cuda-gdb-libs

libtirpc:
	@echo -e "\033[36m----> Building libtirpc\033[0m"
	$(MAKE) -C submodules libtirpc/install

gpu: cuda-gdb
	@echo -e "\033[36m----> Building gpu\033[0m"
	$(MAKE) -C gpu

cpu: libtirpc
	@echo -e "\033[36m----> Building cpu\033[0m"
	$(MAKE) -C cpu

tests:
	@echo -e "\033[36m----> Building test kernels\033[0m"
	$(MAKE) -C tests

install-cpu: bin/cricket-client.so bin/cricket-rpc-server bin/libtirpc.so bin/libtirpc.so.3 bin/tests
	@echo -e "\033[36m----> Copying cpu binaries to build/bin\033[0m"

install: install-cpu bin/cricket
	@echo -e "\033[36m----> Copying to build/bin\033[0m"

bin:
	mkdir bin

bin/tests: bin tests
	ln -s ../tests/bin bin/tests

bin/cricket-client.so: bin
	$(MAKE) -C cpu cricket-client.so
	cp cpu/cricket-client.so bin

bin/cricket-server.so: bin
	$(MAKE) -C cpu cricket-server.so
	mv cpu/cricket-server.so bin/cricket-server.so


bin/cricket-rpc-server: bin
	$(MAKE) -C cpu cricket-rpc-server
	mv cpu/cricket-rpc-server bin/cricket-rpc-server

bin/cricket: bin gpu
	cp gpu/cricket bin

bin/libtirpc.so: bin submodules/libtirpc/install/lib/libtirpc.so
	cp submodules/libtirpc/install/lib/libtirpc.so bin

bin/libtirpc.so.3: bin submodules/libtirpc/install/lib/libtirpc.so.3 libtirpc
	cp submodules/libtirpc/install/lib/libtirpc.so.3 bin

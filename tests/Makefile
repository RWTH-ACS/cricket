#MIT License...
.PHONY: all clean test_apps cpu gpu samples

all: test_apps cpu gpu samples bin

test_apps:
	@echo -e "\033[36m----> Building tests/test_apps\033[0m"
	$(MAKE) -C test_apps

cpu:
	@echo -e "\033[36m----> Building tests/cpu\033[0m"
	$(MAKE) -C cpu

gpu:
	@echo -e "\033[36m----> Building tests/gpu\033[0m"
	$(MAKE) -C gpu

samples:
	@echo -e "\033[36m----> Building tests/samples\033[0m"
	$(MAKE) -C samples

bin: cpu samples test_apps
	mkdir -p bin
	cp cpu/*.test bin
	cp test_apps/*.testapp bin
	cp samples/matrixMul/matrixMul bin
	cp samples/bandwidthTest/bandwidthTest bin
	cp samples/nbody/nbody bin

clean:
	@echo -e "\033[31m----> Cleaning up tests/test_apps\033[0m"
	$(MAKE) -C test_apps clean
	@echo -e "\033[31m----> Cleaning up tests/cpu\033[0m"
	$(MAKE) -C cpu clean
	@echo -e "\033[31m----> Cleaning up tests/gpu\033[0m"
	$(MAKE) -C gpu clean
	@echo -e "\033[31m----> Cleaning up test/samples\033[0m"
	$(MAKE) -C samples clean
	rm -rf bin


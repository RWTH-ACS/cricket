#MIT License...
.PHONY: all clean test_apps cpu gpu samples

ifdef NOSAMPLES
all: test_apps cpu gpu bin
else
all: test_apps cpu gpu samples bin
endif

test_apps:
	@echo -e "\033[36m----> Building tests/test_apps\033[0m"
	$(MAKE) -C test_apps

cpu:
	@echo -e "\033[36m----> Building tests/cpu\033[0m"
	$(MAKE) -C cpu/unit
	$(MAKE) -C cpu/cubin

gpu:
	@echo -e "\033[36m----> Building tests/gpu\033[0m"
	$(MAKE) -C gpu

samples:
	@echo -e "\033[36m----> Building tests/samples\033[0m"
	$(MAKE) -C samples

bin: cpu test_apps
	mkdir -p bin
	cp cpu/unit/*.test bin
	cp test_apps/*.testapp bin
ifndef NOSAMPLES
	cp samples/samples-bin/*.sample bin
endif

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


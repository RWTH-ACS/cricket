#MIT License...
.PHONY: all submodules gpu cpu tests clean

all: gpu cpu tests

clean:
	@echo -e "\033[31m----> Cleaning up gpu\033[0m"
	$(MAKE) -C gpu clean
	@echo -e "\033[31m----> Cleaning up cpu\033[0m"
	$(MAKE) -C cpu clean
	@echo -e "\033[31m----> Cleaning up test kernels\033[0m"
	$(MAKE) -C tests clean

submodules:
	@echo -e "\033[36m----> Building submodules\033[0m"
	$(MAKE) -C submodules

gpu:
	@echo -e "\033[36m----> Building gpu\033[0m"
	$(MAKE) -C gpu

cpu: submodules
	@echo -e "\033[36m----> Building cpu\033[0m"
	$(MAKE) -C cpu

tests:
	@echo -e "\033[36m----> Building test kernels\033[0m"
	$(MAKE) -C tests

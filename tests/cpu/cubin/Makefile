#License...

CC = nvcc
ARCH = sm_75
CFLAGS = -arch=$(ARCH) -cudart shared
LD = nvcc
LDFLAGS = -arch=$(ARCH) -cudart shared -lcuda
BINARY = main
CUBIN = kernel.cubin
FATBIN = kernel.fatbin
SHARED = kernel.so

FILES := main.o

.PHONY: all depend clean 
all : $(BINARY) $(CUBIN) $(FATBIN) $(SHARED)

$(BINARY) : $(FILES)
	$(LD) $(LDFLAGS) -o $@ $< 

%.cubin : %.cu
	$(CC) $(CFLAGS) --cubin -o $@ $<

%.fatbin : %.cu
	$(CC) $(CFLAGS) -o $@ $<

%.o : %.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

%.so : %.cu
	$(CC) $(CFLAGS) --compiler-options '-fPIC' -o $@ $<

clean :
	rm -f *.o *.cubin *.fatbin $(BINARY)



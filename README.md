# cricket

[![pipeline status](https://git.rwth-aachen.de/acs/public/virtualization/cricket-ci/badges/master/pipeline.svg)](https://git.rwth-aachen.de/acs/public/virtualization/cricket-ci/-/commits/master)

Cricket consists of two parts: A virtualization layer for CUDA applications that allows the isolation of CPU and CPU parts by using Remote Procedure Calls and a checkpoint/restart tool for GPU kernels.

# Dependencies
Cricket requires
- CUDA Toolkit (E.g. CUDA 12.1)
- `rpcbind`
- `libcrypto`
- `libtirpc`

libtirpc built as part of the main Makefile.

On the system where the Cricket server should be executed, the appropriate NVIDIA drivers should be installed.

# Building

```
git clone https://github.com/RWTH-ACS/cricket.git
cd cricket && git submodule update --init
LOG=INFO make
```

Environment variables for Makefile:
- `LOG`: Log level. Can be one of `DEBUG`, `INFO`, `WARNING`, `ERROR`.
- `WITH_IB`: If set to `YES` build with Infiniband support.
- `WITH_DEBUG`: Use gcc debug flags for compilation

# Running a CUDA Application
By default Cricket uses TCP/IP as a transport for the Remote Procedure Calls. This enables both remote execution, where server and client execute on different systems and local execution, where server and client execute on the same system.
To support Cricket, the CUDA libraries must be linked dynamically to the CUDA application. For the runtime library, this can be done using the '-cudart shared' flag of `nvcc`. 

The Cricket library has to be preloaded to the CUDA Application.
For starting the server:
```
<path-to-cricket>/bin/cricket-rpc-server [optional rpc id]
```
The client can be started like this:
```
CRICKET_RPCID=[optional rpc id] REMOTE_GPU_ADDRESS=<address-of-server> LD_PRELOAD=<path-to-cricket>/bin/cricket-client.so <cuda-binary>
```

### Example: Running a test application locally
```
/opt/cricket/bin/cricket-rpc-server
```
```
REMOTE_GPU_ADDRESS=127.0.0.1 LD_PRELOAD=/opt/cricket/bin/cricket-client.so /opt/cricket/tests/test_kernel
```

### Example: Running the `nbody` CUDA sample using Cricket on a remote system
Compile the application
```
cd /nfs_share/cuda/samples/5_Simulations/nbody
make NVCCFLAGS="-m64 -cudart shared" GENCODE_FLAGS="-arch=sm_61"
```
Start the Cricket server
```
/opt/cricket/bin/cricket-rpc-server
```
Run the application
```
REMOTE_GPU_ADDRESS=remoteSystem.my-domain.com LD_PRELOAD=/nfs_share/cricket/bin/cricket-client.so /nfs_share/cuda/samples/5_Simulations/nbody/nbody -benchmark
```


# Contributing

## File structue
* **cpu:** The virtualization layer
* **gpu:** The checkpoint/restart tool
* **submodules:** Submodules are located here.
    * **cuda-gdb:** modified GDB for use with CUDA. We mostly need the modified libbfd for gathering information from the CUDA ELF.
    * **libtirpc:** Transport Indepentend Remote Procedure Calls is requried for the virtualization layer-
* **tests:** some synthetic CUDA applications to test cricket.
* **utils:** A Dockerfile for repoducibility and for our CI.

Please agree to the [DCO](DCO.md) by signing off your commits.

## Style Guidelines:
```
set cindent
set tabstop=4
set shiftwidth=4
set expandtab
set cinoptions=(0,:0,l1,t0,L3
match ErrorMsg /\s\+$\| \+\ze\t/
```

This project adheres to the [Linux Kernel Coding Style](https://www.kernel.org/doc/html/v4.10/process/coding-style.html), except when it doesn't.

Etymology: Cricket is an abbreviation for Checkpoint Restart In Cuda KErnels Tool

## Acknowledgments

<p>
    <img src="assets/EN_Funded_by_European_Union_vert_RGB_POS.png#gh-light-mode-only" height="250" alt="Funded by the European Union—NextGenerationEU" />
    <img src="assets/EN_Funded_by_European_Union_vert_RGB_NEG.png#gh-dark-mode-only" height="250" alt="Funded by the European Union—NextGenerationEU" />
    <img src="assets/bmbf_internet_in_farbe_en.jpg" height="250" alt="Sponsored by the Federal Ministry of Education and Research" />
</p>

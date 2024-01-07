# Cricket

[![pipeline status](https://git.rwth-aachen.de/acs/public/virtualization/cricket-ci/badges/master/pipeline.svg)](https://git.rwth-aachen.de/acs/public/virtualization/cricket-ci/-/commits/master)

Cricket is a virtualization layer for CUDA application that enables remote execution and checkpoint/restart without the need to recompile applications.
Cricket isolates CUDA applications from the CUDA APIs by using ONC Remote Procedure Calls.
User code and CUDA APIs are thus executed in separate processes.

![virtualization layer](assets/virt-layer.svg)

For Cricket to be able to insert the virtualization layer, the CUDA application has to link dynamically to the CUDA APIs. For this, you have to pass `-cudart shared` to `nvcc` during linking.

- For experimental pytorch support see [here](docs/pytorch.md).
- For using Cricket from Rust see [here](https://github.com/RWTH-ACS/RPC-Lib).

Supported transports for cudaMemcpy:
- TCP (slow, for pageable memory)
- Infiniband (fast, for pinned memory)
- Shared Memory (fastest, for pinned memory and no remote execution)

# Dependencies
Cricket requires
- CUDA Toolkit (E.g. CUDA 12.1)
- `rpcbind`
- `libcrypto`
- `libtirpc`

libtirpc is built as part of the main Makefile.

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
* **gpu:** experimental in-kernel checkpoint/restart
* **submodules:** Submodules are located here.
    * **cuda-gdb:** modified GDB for use with CUDA. This is only required for in-kernel checkpoint/restart
    * **libtirpc:** Transport Indepentend Remote Procedure Calls is requried for the virtualization layer
* **tests:** various CUDA applications to test cricket.
* **utils:** A Dockerfile for for our CI.s

Please agree to the [DCO](DCO.md) by signing off your commits.

## Publications

Eiling et. al: A virtualization layer for distributed execution of CUDA applications with checkpoint/restart support. Concurrency and Computation: Practice and Experience. 2022. https://doi.org/10.1002/cpe.6474

Eiling et. al: Checkpoint/Restart for CUDA Kernels. In Proceedings of the SC '23 Workshops of The International Conference on High Performance Computing, Network, Storage, and Analysis (SC-W '23). 2023. ACM. https://doi.org/10.1145/3624062.3624254
      
Eiling et. al: GPU Acceleration in Unikernels Using Cricket GPU Virtualization. In Proceedings of the SC '23 Workshops of The International Conference on High Performance Computing, Network, Storage, and Analysis (SC-W '23). 2023. ACM. https://doi.org/10.1145/3624062.3624236

Eiling et. al: An Open-Source Virtualization Layer for CUDA Applications. In Euro-Par 2020: Parallel Processing Workshops. 2021. Lecture Notes in Computer Science, vol 12480. Springer. https://doi.org/10.1007/978-3-030-71593-9_13



## Acknowledgments

<p>
    <img src="assets/EN_Funded_by_European_Union_vert_RGB_POS.png#gh-light-mode-only" height="250" alt="Funded by the European Union—NextGenerationEU" />
    <img src="assets/EN_Funded_by_European_Union_vert_RGB_NEG.png#gh-dark-mode-only" height="250" alt="Funded by the European Union—NextGenerationEU" />
    <img src="assets/bmbf_internet_in_farbe_en.jpg" height="250" alt="Sponsored by the Federal Ministry of Education and Research" />
</p>

# cricket

[![pipeline status](https://git.rwth-aachen.de/niklas.eiling/cricket/badges/master/pipeline.svg)](https://git.rwth-aachen.de/niklas.eiling/cricket/commits/master)

Cricket consists of two parts: A virtualization layer for CUDA applications that allows the isolation of CPU and CPU parts by using Remote Procedure Calls and a checkpoint/restart tool for GPU kernels.

Because of the interdependence of both tools, a separation of both parts is currently not possible.

# File structue
* **cpu:** The virtualization layer
* **gpu:** The checkpoint/restart tool
* **submodules:** Submodules are located here.
    * **cuda-gdb:** modified GDB for use with CUDA. We mostly need the modified libbfd for gathering information from the CUDA ELF.
    * **libtirpc:** Transport Indepentend Remote Procedure Calls is requried for the virtualization layer-
* **tests:** some synthetic CUDA applications to test cricket.
* **utils:** A Dockerfile for repoducibility and for our CI.


Etymology: Cricket is an abbreviation for Checkpoint Restart In Cuda KErnels Tool

# Contributing

Style Guidelines:
```
set cindent
set tabstop=4
set shiftwidth=4
set expandtab
set cinoptions=(0,:0,l1,t0,L3
match ErrorMsg /\s\+$\| \+\ze\t/
```

This project adheres to the [Linux Kernel Coding Style](https://www.kernel.org/doc/html/v4.10/process/coding-style.html), except when it doesn't.

# Cricket pyTorch

Get pytorch sources
```
git clone git@github.com:pytorch/pytorch.git
git checkout v1.13.1
git submodule update --init --recursive
```

patch sources.
- link cudart dynamically when building docker image
- link cudart dynamically when building ATen
- link cudart dynamically when building nccl
- deactivate building for some old cuda versions. (optional)
- add cricket and dependencies to dockerfile
- deactivate compression we do not fully support fatbin compression
- remove compression from nccl as well
```
diff --git a/CMakeLists.txt b/CMakeLists.txt
index e2e1f69457e..f6e5542f341 100644
--- a/CMakeLists.txt
+++ b/CMakeLists.txt
@@ -554,7 +554,6 @@ if(MSVC)
   string(APPEND CMAKE_CUDA_FLAGS " -Xcompiler /w -w")
 endif(MSVC)

-string(APPEND CMAKE_CUDA_FLAGS " -Xfatbin -compress-all")

 if(NOT MSVC)
   string(APPEND CMAKE_CUDA_FLAGS_DEBUG " -g -lineinfo --source-in-ptx")

diff --git a/Dockerfile b/Dockerfile
index 815a9108ce9..53ec7689493 100644
--- a/Dockerfile
+++ b/Dockerfile
@@ -53,7 +53,7 @@ WORKDIR /opt/pytorch
 COPY --from=conda /opt/conda /opt/conda
 COPY --from=submodule-update /opt/pytorch /opt/pytorch
 RUN --mount=type=cache,target=/opt/ccache \
-    TORCH_CUDA_ARCH_LIST="3.5 5.2 6.0 6.1 7.0+PTX 8.0" TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
+    TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 8.0" TORCH_NVCC_FLAGS="-cudart shared --no-compress" \
     CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
     python setup.py install

@@ -93,3 +93,13 @@ WORKDIR /workspace
 FROM official as dev
 # Should override the already installed version from the official-image stage
 COPY --from=build /opt/conda /opt/conda
+RUN apt-get update && apt-get install -y --no-install-recommends \
+        rpcbind \
+        git \
+        automake \
+        libtool \
+        libssl-dev \
+        inetutils-ping \
+        vim \
+        gdb && \
+    rm -rf /var/lib/apt/lists/*
diff --git a/aten/src/ATen/CMakeLists.txt b/aten/src/ATen/CMakeLists.txt
index 3055e290094..4cc14c794b0 100644
--- a/aten/src/ATen/CMakeLists.txt
+++ b/aten/src/ATen/CMakeLists.txt
@@ -458,7 +458,7 @@ if(USE_CUDA AND NOT USE_ROCM)
   endif()
   if($ENV{ATEN_STATIC_CUDA})
     list(APPEND ATen_CUDA_DEPENDENCY_LIBS "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libculibos.a")
-    list(APPEND ATen_CUDA_DEPENDENCY_LIBS "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudart_static.a")
+    list(APPEND ATen_CUDA_DEPENDENCY_LIBS "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudart.so")
   endif($ENV{ATEN_STATIC_CUDA})
 endif()
```
`third_party/nccl/nccl`
```
diff --git a/makefiles/common.mk b/makefiles/common.mk
index 1a1c2b6..c781b39 100644
--- a/makefiles/common.mk
+++ b/makefiles/common.mk
@@ -54,7 +54,7 @@ CXXFLAGS   := -DCUDA_MAJOR=$(CUDA_MAJOR) -DCUDA_MINOR=$(CUDA_MINOR) -fPIC -fvisi                                                                                             
 # Maxrregcount needs to be set accordingly to NCCL_MAX_NTHREADS (otherwise it will cause kernel launch errors)                                                                                
 # 512 : 120, 640 : 96, 768 : 80, 1024 : 60
 # We would not have to set this if we used __launch_bounds__, but this only works on kernels, not on functions.                                                                               
-NVCUFLAGS  := -ccbin $(CXX) $(NVCC_GENCODE) -std=c++11 --expt-extended-lambda -Xptxas -maxrregcount=96 -Xfatbin -compress-all                                                                 
+NVCUFLAGS  := -ccbin $(CXX) $(NVCC_GENCODE) -std=c++11 --expt-extended-lambda -Xptxas -maxrregcount=96 --no-compress -cudart shared                                                           
 # Use addprefix so that we can specify more than one path
 NVLDFLAGS  := -L${CUDA_LIB} -lcudart -lrt
```

Avoid `CMake Error: File /opt/pytorch/build_variables.bzl does not exist.` (https://github.com/pytorch/pytorch/pull/85947):
```
diff --git a/.gitignore b/.gitignore
index 3e6f3831c4c..db6d9c3527e 100644
--- a/.gitignore
+++ b/.gitignore
@@ -214,6 +214,7 @@ build_host_protoc
 build_android
 build_ios
 /build_*
+!/build_variables.bzl
 .build_debug/*
 .build_release/*
 .build_profile/*
```

build pytorch
```
# only necessary when building on an NFS share
EXTRA_DOCKER_BUILD_FLAGS='--storage-opt "overlay.mount_program=/usr/bin/fuse-overlayfs"'

make -f docker.Makefile
```

launch docker container, torch
```
sudo docker run --gpus all --rm -it -v /home/eiling/projects/cricket:/cricket --ipc=host pytorch:latest
REMOTE_GPU_ADDRESS=<cricket server address> LD_PRELOAD=cricket/cpu/cricket-client.so python3

LD_LIBRARY_PATH=/cricket/cpu gdb -x /cricket/tests/gdb_client_cmds python3
(gdb) run /cricket/tests/test_apps/pytorch_minimal.py 
```


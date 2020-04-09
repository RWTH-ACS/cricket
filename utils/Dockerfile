FROM centos:7

RUN yum install -y centos-release-scl
RUN yum install -y clang make bash git gcc autoconf libtool automake
RUN yum install -y ncurses-devel zlib-devel binutils-devel mesa-libGL-devel libvdpau-devel mesa-libEGL-devel rpcbind

ENV LD_LIBRARY_PATH="/usr/local/lib:/usr/local/lib64:${LD_LIBRARY_PATH}"

RUN yum install -y https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-repo-rhel7-10.2.89-1.x86_64.rpm
RUN yum install -y cuda-compiler-10-2 cuda-libraries-dev-10-2 cuda-samples-10-2 --nogpgcheck
RUN ln -s cuda-10.2 /usr/local/cuda

ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

COPY --chown=root .ssh /root/.ssh

WORKDIR /cricket

ENTRYPOINT /bin/bash


FROM centos:8

RUN dnf -y update && \
    dnf install -y epel-release dnf-plugins-core && \
    dnf install -y https://rpms.remirepo.net/enterprise/remi-release-8.rpm && \
    dnf config-manager --set-enabled PowerTools && \
    dnf config-manager --set-enabled remi

RUN dnf install -y make bash git gcc autoconf libtool automake rpcgen \
                   ncurses-devel zlib-devel binutils-devel mesa-libGL-devel \
                   libvdpau-devel mesa-libEGL-devel openssl-devel rpcbind

ENV LD_LIBRARY_PATH="/usr/local/lib:/usr/local/lib64:${LD_LIBRARY_PATH}"

RUN dnf -y install https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-repo-rhel8-10.2.89-1.x86_64.rpm && \
    dnf --refresh -y install cuda-compiler-10-2 cuda-libraries-dev-10-2 cuda-samples-10-2 && \
    ln -s cuda-10.2 /usr/local/cuda

ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

COPY --chown=root .ssh /root/.ssh

WORKDIR /cricket

ENTRYPOINT /bin/bash


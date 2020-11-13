FROM centos:8

LABEL \
	org.label-schema.schema-version = "1.0" \
	org.label-schema.name = "cricket" \
	org.label-schema.license = "MIT" \
	org.label-schema.vendor = "Institute for Automation of Complex Power Systems, RWTH Aachen University" \
	org.label-schema.author.name = "Niklas Eiling" \
	org.label-schema.author.email = "niklas.eiling@eonerc.rwth-aachen.de" \
	org.label-schema.vcs-url = "https://git.rwth-aachen.de/niklas.eiling/cricket"

RUN dnf -y update && \
    dnf install -y epel-release dnf-plugins-core && \
    dnf install -y https://rpms.remirepo.net/enterprise/remi-release-8.rpm && \
    dnf config-manager --set-enabled PowerTools && \
    dnf config-manager --set-enabled remi

RUN dnf install -y make bash git gcc autoconf libtool automake rpcgen \
                   ncurses-devel zlib-devel binutils-devel mesa-libGL-devel \
                   libvdpau-devel mesa-libEGL-devel openssl-devel rpcbind \
                   texinfo bison flex python3 which libibverbs

ENV LD_LIBRARY_PATH="/usr/local/lib:/usr/local/lib64:${LD_LIBRARY_PATH}"

RUN dnf -y config-manager --add-repo http://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo && \
    dnf --refresh -y install cuda-compiler-11-1 cuda-libraries-devel-11-1 cuda-samples-11-1 && \
    ln -s cuda-11.1 /usr/local/cuda

ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

#COPY --chown=root .ssh /root/.ssh

WORKDIR /cricket

ENTRYPOINT /bin/bash


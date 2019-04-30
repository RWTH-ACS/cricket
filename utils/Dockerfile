FROM centos:7

RUN yum install -y centos-release-scl
RUN yum install -y clang make bash git gcc
RUN yum install -y ncurses-devel.x86_64 zlib-devel.x86_64

ENV LD_LIBRARY_PATH /usr/local/lib:/usr/local/lib64

WORKDIR /cricket

ENTRYPOINT bash


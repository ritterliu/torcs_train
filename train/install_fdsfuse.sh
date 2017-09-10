#!/bin/bash

# This version is tested against ubuntu 14.04 and 16.04.
# For poco and fds-sdk-cpp, it is compiled from scratch, it takes some time.
# print COMMANDS as they are executed
set -x
 
apt-get install -y \
    libfuse-dev \
    libcurl4-gnutls-dev \
    libssl-dev \
    libxml2-dev

wget -q -O - http://cnbj1.fds.api.xiaomi.com/docker-image-package/fuse/poco-1.7.7-all.tar.gz | tar -xzvf - && \
    cd poco-1.7.7-all && \
    ./configure && \
    make && \
    make install && \
    cd .. && \
    rm -rf poco-1.7.7-all

export DEPS_CFLAGS=" -D_FILE_OFFSET_BITS=64 -I/usr/include/libxml2 -I/usr/include/fuse -I/usr/local/include/galaxy/fds  "
export DEPS_LIBS=" -pthread -lfuse -lrt -ldl -lcurl -lxml2 -lcrypto -lPocoNet -lPocoNetSSL -lPocoFoundation -lPocoJSON -lgalaxy-fds-sdk-cpp  "
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib"

wget -q -O - http://cnbj1.fds.api.xiaomi.com/docker-image-package/fds-sdk-cpp-release.tar.gz | tar -xzvf - && \
    cd fds-sdk-cpp-release && \
    ./install.sh && \
    cd .. && \
    rm -rf fds-sdk-cpp-release

wget -q -o - http://cnbj1.fds.api.xiaomi.com/docker-image-package/fdsfuse_0.0.2-1_amd64.deb && \
    dpkg -i fdsfuse_0.0.2-1_amd64.deb  && \
    rm -rf fdsfuse_0.0.2-1_amd64.deb
 
mkdir /fds
mkdir /fdscache
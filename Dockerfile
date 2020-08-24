FROM ubuntu:18.04
LABEL maintainer "Jaewoo Kim <jwkimrhkgkr@gmail.com>"

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

WORKDIR /app/build
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git
RUN git submodule update  --init
RUN sudo apt install software-properties-common && \
    sudo add-apt-repository ppa:ubuntu-toolchain-r/test && \
    sudo apt install gcc-9 g++-9 && \
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90 && \
    sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 90 && \
RUN cmake .. && \
    make  && \
    make install
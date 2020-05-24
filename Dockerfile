FROM ubuntu:18.04
LABEL maintainer "Chris Ohk <utilforever@gmail.com>"

RUN apt-get update && apt-get install -y \
    build-essential \
    libcurl4-openssl-dev \
    zlib1g-dev \
    cmake \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

WORKDIR /app/build
RUN cmake .. && \
    make -j "$(nproc)" && \
    make install && \
    bin/UnitTests
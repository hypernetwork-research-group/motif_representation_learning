ARG PLAT="linux/amd64"

FROM --platform=${PLAT} ubuntu:24.04

ARG PLAT="linux/amd64"

WORKDIR /app

RUN apt update
RUN apt install -y wget 
RUN apt install -y gcc-13

COPY . /app

RUN chmod +x setup.sh
RUN bash setup.sh ${PLAT}

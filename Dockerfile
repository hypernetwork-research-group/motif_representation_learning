FROM --platform="linux/amd64" ubuntu:24.04

WORKDIR /app

RUN apt update
RUN apt install -y wget
RUN apt install -y gcc-13

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /opt/miniconda.sh
RUN bash /opt/miniconda.sh -b -p /opt/miniconda

COPY . /app

RUN chmod +x setup.sh
RUN bash setup.sh

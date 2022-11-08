ARG TORCH_VERSION=latest

FROM pytorch/pytorch:${TORCH_VERSION}

USER root
RUN apt-get -qq -y update && \
    apt-get -qq -y upgrade && \
    apt-get -qq -y install \
        wget \
        curl \
        git \
        make \
        sudo

COPY docker/requirements.txt .
COPY . .

RUN python3 -m pip install -r requirements.txt

EXPOSE 8888

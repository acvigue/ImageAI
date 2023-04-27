ARG PYTHON_VERSION="3.11.3"
FROM python:${PYTHON_VERSION}-slim-buster

LABEL mantainer="Aiden Vigue <aiden@vigue.me>"

RUN apt-get update \
    && apt-get install -y \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        yasm \
        pkg-config \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavformat-dev \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

ARG OPENCV_VERSION="4.7.0"
ARG SYSTEM_CORES="4"
RUN cp /usr/bin/make /usr/bin/make.bak && \
    echo "make.bak --jobs=${SYSTEM_CORES} \$@" > /usr/bin/make && \
    pip install -v opencv-python==${OPENCV_VERSION} && \
    mv /usr/bin/make.bak /usr/bin/make

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 8000

CMD [ "python3", "-m" , "sanic", "index:app"]

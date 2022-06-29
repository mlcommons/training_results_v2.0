# A image for building paddle binaries

# build docker image
# docker build -t mlperf_paddle_bert:latest .

# run a container
# docker run \
# --ulimit memlock=-1:-1 --net=host --cap-add=IPC_LOCK --device=/dev/infiniband/ --ipc=host \
# --rm -it \
# -v ${IPUOF_CONFIG_PATH}:/ipuof.conf \
# -e IPUOF_CONFIG_PATH=/ipuof.conf \
# mlperf_paddle_bert:latest bash

FROM graphcore/poplar-extbaidu:latest

# ENV variables
ARG WITH_AVX
ENV HOME /root
ENV WITH_AVX=${WITH_AVX:-ON}

RUN apt-get update && apt-get install -y apt-utils
RUN ln -snf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata
RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa && add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update && apt-get install -y curl wget vim git unzip unrar tar xz-utils libssl-dev bzip2 gzip make libgcc-s1 sudo openssh-server \
    coreutils ntp language-pack-zh-hans python-qt4 libsm6 libxext6 libxrender-dev libgl1-mesa-glx libsqlite3-dev libopenblas-dev \
    bison graphviz libjpeg-dev zlib1g zlib1g-dev automake locales swig net-tools libtool module-init-tools numactl libnuma-dev \
    openssl libffi-dev pciutils libblas-dev gfortran libblas3 liblapack-dev liblapack3 default-jre screen tmux gdb lldb gcc g++
RUN apt-get update && apt-get install -y rdma-core librdmacm1
RUN apt-get update && apt-get install libspdlog-dev

# clean apt cache
RUN apt-get clean -y

# Locales
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
# Fix locales to en_US.UTF-8
RUN localedef -i en_US -f UTF-8 en_US.UTF-8

# Downgrade gcc&&g++
WORKDIR /usr/bin
# RUN wget https://raw.githubusercontent.com/PaddlePaddle/Paddle/develop/tools/dockerfile/build_scripts/install_gcc.sh -O /root/install_gcc.sh
RUN wget -q --no-proxy https://paddle-ci.gz.bcebos.com/gcc-8.2.0.tar.xz
RUN tar -xf gcc-8.2.0.tar.xz && \
    cd gcc-8.2.0 && \
    unset LIBRARY_PATH CPATH C_INCLUDE_PATH PKG_CONFIG_PATH CPLUS_INCLUDE_PATH INCLUDE && \
    wget http://gcc.gnu.org/pub/gcc/infrastructure/gmp-6.1.0.tar.bz2 && \
    wget http://gcc.gnu.org/pub/gcc/infrastructure/mpfr-3.1.4.tar.bz2 && \
    wget http://gcc.gnu.org/pub/gcc/infrastructure/mpc-1.0.3.tar.gz && \
    wget http://gcc.gnu.org/pub/gcc/infrastructure/isl-0.18.tar.bz2 && \
    ./contrib/download_prerequisites && \
    cd .. && mkdir temp_gcc82 && cd temp_gcc82 && \
    ../gcc-8.2.0/configure --prefix=/usr/local/gcc-8.2 --enable-threads=posix --disable-checking --disable-multilib && \
    make -j`nproc` && make install
RUN cd .. && rm -rf temp_gcc82
RUN cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /usr/lib/x86_64-linux-gnu/libstdc++.so.6.bak  && rm -f /usr/lib/x86_64-linux-gnu/libstdc++.so.6 && \
    ln -s /usr/local/gcc-8.2/lib64/libgfortran.so.5 /usr/lib/x86_64-linux-gnu/libstdc++.so.5 && \
    ln -s /usr/local/gcc-8.2/lib64/libstdc++.so.6 /usr/lib/x86_64-linux-gnu/libstdc++.so.6 && \
    cp /usr/local/gcc-8.2/lib64/libstdc++.so.6.0.25 /usr/lib/x86_64-linux-gnu
RUN cp gcc gcc.bak && cp g++ g++.bak && rm gcc && rm g++
RUN ln -s /usr/local/gcc-8.2/bin/gcc /usr/local/bin/gcc
RUN ln -s /usr/local/gcc-8.2/bin/g++ /usr/local/bin/g++
RUN ln -s /usr/local/gcc-8.2/bin/gcc /usr/bin/gcc
RUN ln -s /usr/local/gcc-8.2/bin/g++ /usr/bin/g++
ENV PATH=/usr/local/gcc-8.2/bin:$PATH

# install cmake
WORKDIR /home
RUN wget -q https://cmake.org/files/v3.16/cmake-3.16.0-Linux-x86_64.tar.gz && tar -zxvf cmake-3.16.0-Linux-x86_64.tar.gz && rm cmake-3.16.0-Linux-x86_64.tar.gz
ENV PATH=/home/cmake-3.16.0-Linux-x86_64/bin:$PATH

# conda 4.9.2
WORKDIR /opt
ARG CONDA_FILE=Miniconda3-py37_4.9.2-Linux-x86_64.sh
RUN cd /opt && wget -q https://repo.anaconda.com/miniconda/${CONDA_FILE} && chmod +x ${CONDA_FILE}
RUN mkdir /opt/conda && ./${CONDA_FILE} -b -f -p "/opt/conda" && rm -rf ${CONDA_FILE}
ENV PATH=/opt/conda/bin:${PATH}
RUN conda init bash && conda install -n base jupyter jupyterlab

# git credential to skip password typing
RUN git config --global credential.helper store

# install pylint and pre-commit
RUN /opt/conda/bin/pip install pre-commit pylint pytest astroid isort protocol PyGithub

# install Paddle requirement
RUN wget https://raw.githubusercontent.com/PaddlePaddle/Paddle/develop/python/requirements.txt -O /root/requirements.txt
RUN /opt/conda/bin/pip install -r /root/requirements.txt && \
    rm -rf /root/requirements.txt

RUN wget https://raw.githubusercontent.com/PaddlePaddle/Paddle/develop/python/unittest_py/requirements.txt -O /root/requirements.txt
RUN /opt/conda/bin/pip install -r /root/requirements.txt && rm -rf /root/requirements.txt

# Older versions of patchelf limited the size of the files being processed and were fixed in this pr.
# https://github.com/NixOS/patchelf/commit/ba2695a8110abbc8cc6baf0eea819922ee5007fa
# So install a newer version here.
RUN wget -q https://paddle-ci.cdn.bcebos.com/patchelf_0.10-2_amd64.deb && \
    dpkg -i patchelf_0.10-2_amd64.deb

# ccache 3.7.9
RUN wget https://paddle-ci.gz.bcebos.com/ccache-3.7.9.tar.gz && \
    tar xf ccache-3.7.9.tar.gz && mkdir /usr/local/ccache-3.7.9 && cd ccache-3.7.9 && \
    ./configure -prefix=/usr/local/ccache-3.7.9 && \
    make -j8 && make install && \
    ln -s /usr/local/ccache-3.7.9/bin/ccache /usr/local/bin/ccache

# clang-form 3.8.0
RUN wget https://paddle-ci.cdn.bcebos.com/clang+llvm-3.8.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz && \
    tar xf clang+llvm-3.8.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz && cd clang+llvm-3.8.0-x86_64-linux-gnu-ubuntu-16.04 && \
    cp -r * /usr/local && cd .. && rm -rf clang+llvm-3.8.0-x86_64-linux-gnu-ubuntu-16.04 && rm -rf clang+llvm-3.8.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz

# install Paddlepaddle(IPU)
WORKDIR /opt
# RUN git clone -b mlperf_v2.0 https://github.com/graphcore/Paddle-fork.git Paddle
RUN git clone https://github.com/PaddlePaddle/Paddle.git
WORKDIR /opt/Paddle
RUN git checkout b2d8f6dfb63a996d793b91771b43948c0c48f6ca
RUN ls
RUN mkdir -p build
RUN cmake -DPYTHON_EXECUTABLE=/opt/conda/bin/python3.7 -DCMAKE_INSTALL_PREFIX="install" \
    -DWITH_PYTHON=ON -DWITH_GLOO=ON -DWITH_IPU=ON -DPOPLAR_DIR=/opt/poplar \
    -DON_INFER=ON \
    -DPOPART_DIR=/opt/popart -G "Unix Makefiles" -H`pwd` -B`pwd`/build
RUN cmake --build `pwd`/build --config Release --target paddle_python -j$(nproc)
RUN pip3 install build/python/dist/* --force-reinstall
# RUN rm -rf /opt/Paddle/build

# install requirements for mlperf
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

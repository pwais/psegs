FROM oarphpy/full:0.1.1

## Include banner
COPY docker/bashrc /etc/bash.bashrc
RUN chmod a+rwx /etc/bash.bashrc

RUN \
  apt-get update && \
  apt-get install -y \
    git \
    wget

# # Pytorch3d
# RUN \
#   pip3 install torch==1.9.0+cu111 torchvision==0.9.2+cu111 torchaudio==0.8.2 && \
#   pip3 install pytorch3d==0.6.0 -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu111_pyt190/download.html

# # open3d
# RUN apt-get install ffmpeg libsm6 libxext6 libc++-dev sudo git -y
# RUN apt-get install -y build-essential cmake ccache
# RUN cd /opt && \
#   git clone https://github.com/isl-org/Open3D.git && \
#   cd Open3D && \
#   ./util/install_deps_ubuntu.sh assume-yes && \
#   mkdir -p build && cd build && \
#   cmake \
#     -DBUILD_CUDA_MODULE=1 \
#     -DBUILD_COMMON_CUDA_ARCHS=1 \
#     -DENABLE_HEADLESS_RENDERING=1 \
#     -DBUILD_WEBRTC=1 \
#     -DCMAKE_BUILD_TYPE=Release \
#     .. && \
#   make -j $(nproc)

# # RUN pip3 install open3d==0.13.0
# # RUN \
# #   ln -s \
# #     /usr/local/cuda-11.1/targets/x86_64-linux/lib/libcusolver.so.11 \
# #     $(python3 -c "import open3d as x; print(x.__path__[0])")/cuda/libcusolver.so.10

# # ENV LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:/usr/local/cuda-11.1/targets/x86_64-linux/lib/
# RUN python3 -c ''

# pycolmap and pycolmap: TODO use a wheel
## COLMAP
RUN \
  apt-get install -y \
    git \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev
RUN \
  cd /opt && \
    (git clone --depth 1 --branch 3.8 https://github.com/colmap/colmap || true) && \
    cd colmap && \
    mkdir -p build && cd build && \
    cmake .. -GNinja -DCUDA_ENABLED=OFF && \
    ninja && \
    ninja install && \
    colmap -h
RUN \
  cd /opt && \
  git clone --recursive https://github.com/colmap/pycolmap.git && \
  cd pycolmap && \
  git checkout 401f82658cdad1e8b657c77381863f9e261c7c3c && \
  echo "Hack: build with debug symbols else segfaults on list comps python 3.10" && \
  sed -i -e "s/cfg = 'Debug'/cfg = 'RelWithDebInfo' # /g" setup.py && \
  pip3 install -v -e .

COPY requirements.txt /tmp/requirements.txt
RUN \
  pip3 install -r /tmp/requirements.txt && \
  rm /tmp/requirements.txt

RUN mkdir -p /opt/psegs
COPY . /opt/psegs
WORKDIR /opt/psegs
ENV PYTHONPATH $PYTHONPATH:/opt/psegs

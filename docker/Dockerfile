FROM oarphpy/full:0.1.0

## Include banner
COPY docker/bashrc /etc/bash.bashrc
RUN chmod a+rwx /etc/bash.bashrc

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

# Pytorch3d
RUN \
  pip3 install torch torchvision torchaudio &&
  pip3 install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu102_pyt190/download.html

RUN \
  apt-get update && \
  apt-get install -y \
    git \
    wget

RUN mkdir -p /opt/psegs
COPY . /opt/psegs
WORKDIR /opt/psegs

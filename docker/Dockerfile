FROM oarphpy/full:0.0.4

## Include banner
COPY docker/bashrc /etc/bash.bashrc
RUN chmod a+rwx /etc/bash.bashrc

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

RUN \
  apt-get update && \
  apt-get install -y \
    git \
    wget

RUN mkdir -p /opt/psegs
COPY . /opt/psegs
WORKDIR /opt/psegs

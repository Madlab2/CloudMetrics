FROM python:3.9

# Install system dependencies
RUN apt-get update &&  apt-get install -y \
  libgl1-mesa-glx \
  python3-pip \
  curl

# non root user
ARG USERNAME=ros
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# interactive widgets in jupyter, numpy, open3d, ...
RUN pip install --upgrade --no-cache-dir ipympl \
  pandas \
  matplotlib \
  numpy \
  tqdm \
  open3d \
  laspy \
  pyntcloud
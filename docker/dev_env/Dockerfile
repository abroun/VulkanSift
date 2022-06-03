FROM nvidia/cuda:11.7.0-devel-ubuntu22.04
ARG USER_ID
ARG USER_NAME

# Install deps
RUN apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y  \
    wget unzip git less curl unzip build-essential cmake cmake-curses-gui \
    apt-transport-https \
    ca-certificates \
    gnupg \
    libopencv-dev \
    libx11-dev \
    gdb

RUN mkdir /vulkan
WORKDIR /vulkan
RUN wget https://sdk.lunarg.com/sdk/download/1.3.211.0/linux/vulkansdk-linux-x86_64-1.3.211.0.tar.gz
RUN tar xf vulkansdk-linux-x86_64-1.3.211.0.tar.gz

# NOTE: Can't find a generic libnvidia-gl package so need to match version
# the version fo the base CUDA container.
RUN apt-get update && apt-get install -y libnvidia-gl-515

# Gui deps
RUN apt-get update && apt-get install -y xterm

WORKDIR /src

# Setup username and id so that it matches host machine (otherwise files will be written as root)
RUN useradd -u ${USER_ID} ${USER_NAME}
ENV USER=${USER_NAME}
ENV USER=${USER_NAME}
ENV HOME=/home/${USER_NAME}
RUN mkdir -p /home/${USER_NAME}
RUN usermod -a -G video ${USER_NAME}
RUN chown ${USER_NAME}:${USER_NAME} /home/${USER_NAME}
USER ${USER_NAME}

ENV VULKAN_SDK=/vulkan/1.3.211.0/x86_64
ENV PATH=$VULKAN_SDK/bin:$PATH
ENV LD_LIBRARY_PATH=$VULKAN_SDK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
ENV VK_LAYER_PATH=$VULKAN_SDK/etc/vulkan/explicit_layer.d
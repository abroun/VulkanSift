#!/usr/bin/env bash

USER_ID=`id -u`

IMAGE_TAG=env/vulkan_sift
CONTAINER_NAME_ENV=vulkan_sift_env

# Create the docker environment for building Vulkan SIFT
docker build -t $IMAGE_TAG docker/dev_env/. \
    --build-arg "USER_ID=${USER_ID}" --build-arg "USER_NAME=${USER}"

EXTRA_ARGS=
EXTRA_ARGS+=" -v /media/datasets:/media/datasets "

# Start the command line
docker run --rm -it \
    --name ${CONTAINER_NAME_ENV} \
    -e DISPLAY=${DISPLAY} \
    -v $(pwd):/src \
    -v /tmp:/tmp \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    ${EXTRA_ARGS} \
    --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=all \
    --gpus all $IMAGE_TAG /bin/bash

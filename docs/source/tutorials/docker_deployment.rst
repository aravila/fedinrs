.. _docker-deployment:

********************************************
Deploy FedLab Process in a Docker Container
********************************************

Why docker?
============================

The communication APIs of **FedLab** is built on `torch.distributed <https://pytorch.org/docs/stable/distributed.html>`_. In cross-process scene, when multiple **FedLab** processes are deployed on the same machine, GPU memory buckets will be created automatically however which are not used in our framework. We can start the **FedLab** processes in different docker containers to avoid triggering GPU memory buckets (to save GPU memory).

Setup docker environment
==========================

In this section, we introduce how to setup a docker image for **FedLab** program. Here we provide the Dockerfile for building a FedLab image. Our FedLab environment is based on PytTorch. Therefore, we just need install **FedLab** on the provided PytTorch image.

Dockerfile:

.. code-block:: shell-session

    # This is an example of fedlab installation via Dockerfile

    # replace the value of TORCH_CONTAINER with pytorch image that satisfies your cuda version
    # you can find it in https://hub.docker.com/r/pytorch/pytorch/tags
    ARG TORCH_CONTAINER=1.5-cuda10.1-cudnn7-runtime

    FROM pytorch/pytorch:${TORCH_CONTAINER}

    RUN pip install --upgrade pip \
        & pip uninstall -y torch torchvision  \
        & conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ \
        & conda config --set show_channel_urls yes \
        & mkdir /root/tmp/

    # replace with the correct install command, which you can find in https://pytorch.org/get-started/previous-versions/
    RUN conda install -y pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch 

    # pip install fedlab
    RUN TMPDIR=/root/tmp/ pip install -i https://pypi.mirrors.ustc.edu.cn/simple/ fedlab



Dockerfile for different platforms
==================================

The steps of modifying Dockerfile for different platforms:

- **Step 1:** Find an appropriate base pytorch image for your platform from dockerhub https://hub.docker.com/r/pytorch/pytorch/tags. Then, replace the value of `TORCH_CONTAINER` in demo dockerfile.

- **Step 2:** To install specific PyTorch version, you need to choose a correct install command, which can be find in https://pytorch.org/get-started/previous-versions/. Then, modify the 16-th command in demo dockerfile.

- **Step 3:** Build the images for your own platform by running the command below in the dir of Dockerfile.

  .. code-block:: shell-session

      $ docker build -t image_name .

.. warning::
  
    Using "--gpus all" and "--network=host" when start a docker container:

    .. code-block:: shell-session

        $ docker run -itd --gpus all --network=host b23a9c46cd04(image name) /bin/bash

    If you are not in China area, it is ok to remove line 11,12 and "-i https://pypi.mirrors.ustc.edu.cn/simple/" in line 19.

- **Finally:** Run your FedLab process in the different started containers.

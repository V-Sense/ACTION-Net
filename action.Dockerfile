from nvidia/cuda:11.0-cudnn8-runtime-ubuntu18.04

run apt-get update && apt-get install -y \
    python3\
    python3-pip \
    git\
    ffmpeg\
    && rm -rf /var/lib/apt/lists/*
run pip3 install --upgrade pip
run pip3 install --upgrade cython
run pip3 install \
    torch==1.6.0\
    torchvision==0.7.0\
    scipy\
    pillow==6.2.1\
    sklearn\
    tqdm\
    torchsummary\
    matplotlib\
    opencv-python-headless\
    pandas\
    scikit-image\
    'git+https://github.com/facebookresearch/fvcore'\
    av\
    psutil\
    tensorboardX\
    ptflops


workdir /home/zhengwei
cmd ["/bin/bash"]

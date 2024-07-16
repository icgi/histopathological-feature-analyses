FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /workdir

COPY . .

# Install dependencies
RUN apt-get update && apt-get install --yes \
    build-essential \
    tmux \
    vim

# Install python packages
RUN python -m pip install --upgrade pip && python -m pip install --no-cache-dir \
    timm==0.9.8 \
    opencv_python_headless==4.10.0.82 \
    polars==0.20.31

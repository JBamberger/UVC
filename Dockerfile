FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel

WORKDIR /workspace

# Required for OpenCV python package
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get update && apt-get install git  -y

COPY requirements.txt ./
RUN pip install -r requirements.txt

# copy the code into  the container
COPY . .

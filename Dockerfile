FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

COPY . /home/appuser
WORKDIR /home/appuser
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
# Below is for https://github.com/IDEA-Research/Grounded-Segment-Anything/issues/53
ENV CUDA_HOME /usr/local/cuda-11.6/
RUN python -m pip install -e segment_anything
RUN python -m pip install -e GroundingDINO
RUN pip install --upgrade diffusers[torch]
RUN pip install opencv-python pycocotools matplotlib onnxruntime onnx ipykernel

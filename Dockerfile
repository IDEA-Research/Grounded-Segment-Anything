FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

COPY . /home/appuser
WORKDIR /home/appuser
RUN apt-get update
RUN python -m pip install -e segment_anything
RUN python -m pip install -e GroundingDINO
RUN pip install --upgrade diffusers[torch]
RUN pip install opencv-python pycocotools matplotlib onnxruntime onnx ipykernel

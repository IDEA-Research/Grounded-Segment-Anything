# Grounded-Segment-Anything
Marrying Grounding DINO with Segment Anything: Detect And Segment Anything !


## Highlight
- Detect and Segment everything with Language!


## Catelog
- [x] GroundingDINO + Segment-Anything Demo
- [ ] Gradio Demo

## Installation
The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

Install Segment Anything:

```bash
python -m pip install -e segment-anything
```

Install GroundingDINO:

```bash
python -m pip install -e GroundingDINO
```

The following optional dependencies are necessary for mask post-processing, saving masks in COCO format, the example notebooks, and exporting the model in ONNX format. `jupyter` is also required to run the example notebooks.
```
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

More details can be found in [installation segment anything](https://github.com/facebookresearch/segment-anything#installation) and [installation GroundingDINO](https://github.com/IDEA-Research/GroundingDINO#install)


## Demo

## Paint by Example: Exemplar-based Image Editing with Diffusion Models

:grapes: [[Official Project Page](https://github.com/Fantasy-Studio/Paint-by-Example)] &nbsp; :apple:[[Official Online Demo](https://huggingface.co/spaces/Fantasy-Studio/Paint-by-Example)]

<div align="center">

![](https://github.com/Fantasy-Studio/Paint-by-Example/blob/main/figure/teaser.png?raw=True)

</div>

## Abstract

> Language-guided image editing has achieved great success recently. In this paper, for the first time, we investigate exemplar-guided image editing for more precise control. We achieve this goal by leveraging self-supervised training to disentangle and re-organize the source image and the exemplar. However, the naive approach will cause obvious fusing artifacts. We carefully analyze it and propose an information bottleneck and strong augmentations to avoid the trivial solution of directly copying and pasting the exemplar image. Meanwhile, to ensure the controllability of the editing process, we design an arbitrary shape mask for the exemplar image and leverage the classifier-free guidance to increase the similarity to the exemplar image. The whole framework involves a single forward of the diffusion model without any iterative optimization. We demonstrate that our method achieves an impressive performance and enables controllable editing on in-the-wild images with high fidelity.

## Table of Contents
- [Installation](#installation)
- [Paint-By-Example Demos](#paint-by-example-demos)
  - [Diffuser Demo](#paintbyexample-diffuser-demos)
  - [PaintByExample with SAM](#paintbyexample-with-sam)


## TODO
- [x] PaintByExample Diffuser Demo
- [x] PaintByExample with SAM
- [ ] PaintByExample with GroundingDINO
- [ ] PaintByExample with Grounded-SAM

## Installation
We're using PaintByExample with diffusers, install diffusers as follows:
```bash
pip install diffusers==0.16.1
```
Then install Grounded-SAM follows [Grounded-SAM Installation](https://github.com/IDEA-Research/Grounded-Segment-Anything#installation) for some extension demos.

## Paint-By-Example Demos
Here we provide the demos for `PaintByExample`


### PaintByExample Diffuser Demos
```python
cd playground/PaintByExample
python paint_by_example.py
```
**Notes:** set `cache_dir` to save the pretrained weights to specific folder. The paint result will be save as `paint_by_example_demo.jpg`:

<div align="center">

| Input Image | Mask | Example Image | Inpaint Result |
|:----:|:----:|:----:|:----:|
| ![](https://github.com/IDEA-Research/detrex-storage/blob/main/assets/grounded_sam/paint_by_example/input_image.png?raw=true) | ![](https://github.com/IDEA-Research/detrex-storage/blob/main/assets/grounded_sam/paint_by_example/mask.png?raw=true) | <div style="text-align: center"> <img src="https://github.com/IDEA-Research/detrex-storage/blob/main/assets/grounded_sam/paint_by_example/example_image.jpg?raw=true" width=55%></div> | ![](https://github.com/IDEA-Research/detrex-storage/blob/main/assets/grounded_sam/paint_by_example/paint_by_example_demo.jpg?raw=true) |
| ![](https://github.com/IDEA-Research/detrex-storage/blob/main/assets/grounded_sam/paint_by_example/input_image.png?raw=true) | ![](https://github.com/IDEA-Research/detrex-storage/blob/main/assets/grounded_sam/paint_by_example/mask.png?raw=true) | <div style="text-align: center"> <img src="https://github.com/IDEA-Research/detrex-storage/blob/main/assets/grounded_sam/paint_by_example/pomeranian_example.jpg?raw=true" width=55%></div> | ![](https://github.com/IDEA-Research/detrex-storage/blob/main/assets/grounded_sam/paint_by_example/paint_by_pomeranian_demo.jpg?raw=true) |

</div>

### PaintByExample with SAM

In this demo, we did inpaint task by:
1. Generate mask by SAM with prompt (box or point)
2. Inpaint with mask and example image

```python
cd playground/PaintByExample
python sam_paint_by_example.py
```
**Notes:** We set a more `num_inference_steps` (like 200 to 500) to get higher quality image. And we've found that the mask region can influence a lot on the final result (like a panda can not be well inpainted with a region like dog). It needed to have more test on it.

| Input Image | SAM Output | Example Image | Inpaint Result |
|:----:|:----:|:----:|:----:|
| ![](https://github.com/IDEA-Research/detrex-storage/blob/main/assets/grounded_sam/paint_by_example/input_image.png?raw=true) | ![](https://github.com/IDEA-Research/detrex-storage/blob/main/assets/grounded_sam/paint_by_example/demo_with_point_prompt.png?raw=true) | <div style="text-align: center"> <img src="https://github.com/IDEA-Research/detrex-storage/blob/main/assets/grounded_sam/paint_by_example/labrador_example.jpg?raw=true" width=55%></div> | ![](https://github.com/IDEA-Research/detrex-storage/blob/main/assets/grounded_sam/paint_by_example/sam_paint_by_example_demo.jpg?raw=true) |


## RePaint: Inpainting using Denoising Diffusion Probabilistic Models

:grapes: [[Official Project Page](https://github.com/andreas128/RePaint)]

<div align="center">

![](https://user-images.githubusercontent.com/11280511/150803812-a4729ef8-6ad4-46aa-ae99-8c27fbb2ea2e.png)

</div>

## Abstract

> Free-form inpainting is the task of adding new content to an image in the regions specified by an arbitrary binary mask. Most existing approaches train for a certain distribution of masks, which limits their generalization capabilities to unseen mask types. Furthermore, training with pixel-wise and perceptual losses often leads to simple textural extensions towards the missing areas instead of semantically meaningful generation. In this work, we propose RePaint: A Denoising Diffusion Probabilistic Model (DDPM) based inpainting approach that is applicable to even extreme masks. We employ a pretrained unconditional DDPM as the generative prior. To condition the generation process, we only alter the reverse diffusion iterations by sampling the unmasked regions using the given image information. Since this technique does not modify or condition the original DDPM network itself, the model produces highquality and diverse output images for any inpainting form. We validate our method for both faces and general-purpose image inpainting using standard and extreme masks. RePaint outperforms state-of-the-art Autoregressive, and GAN approaches for at least five out of six mask distributions.


## Table of Contents
- [Installation](#installation)
- [Repaint Demos](#repaint-demos)
  - [Diffuser Demo](#repaint-diffuser-demos)


## TODO
- [x] RePaint Diffuser Demo
- [ ] RePaint with SAM
- [ ] RePaint with GroundingDINO
- [ ] RePaint with Grounded-SAM

## Installation
We're using PaintByExample with diffusers, install diffusers as follows:
```bash
pip install diffusers==0.16.1
```
Then install Grounded-SAM follows [Grounded-SAM Installation](https://github.com/IDEA-Research/Grounded-Segment-Anything#installation) for some extension demos.

## RePaint Demos
Here we provide the demos for `RePaint`


### RePaint Diffuser Demos
```python
cd playground/RePaint
python repaint.py
```
**Notes:** set `cache_dir` to save the pretrained weights to specific folder. The paint result will be save as `repaint_demo.jpg`:

<div align="center">

| Input Image | Mask | Inpaint Result |
|:----:|:----:|:----:|
| ![](https://github.com/IDEA-Research/detrex-storage/blob/main/assets/grounded_sam/repaint/celeba_hq_256.png?raw=true) | ![](https://github.com/IDEA-Research/detrex-storage/blob/main/assets/grounded_sam/repaint/mask_256.png?raw=true) | ![](https://github.com/IDEA-Research/detrex-storage/blob/main/assets/grounded_sam/repaint/repaint_demo.jpg?raw=true) |


</div>



## LaMa: Resolution-robust Large Mask Inpainting with Fourier Convolutions

:grapes: [[Official Project Page](https://advimman.github.io/lama-project/)] &nbsp; :apple:[[LaMa Cleaner](https://github.com/Sanster/lama-cleaner)]

We use the highly organized code [lama-cleaner](https://github.com/Sanster/lama-cleaner) to simplify the demo code for users.

<div align="center">

![](https://raw.githubusercontent.com/senya-ashukha/senya-ashukha.github.io/master/projects/lama_21/ezgif-4-0db51df695a8.gif)

</div>

## Abstract

> Modern image inpainting systems, despite the significant progress, often struggle with large missing areas, complex geometric structures, and high-resolution images. We find that one of the main reasons for that is the lack of an ef-fective receptive field in both the inpainting network andthe loss function. To alleviate this issue, we propose anew method called large mask inpainting (LaMa). LaM ais based on: a new inpainting network architecture that uses fast Fourier convolutions, which have the image-widereceptive field
a high receptive field perceptual loss; large training masks, which unlocks the potential ofthe first two components. Our inpainting network improves the state-of-the-art across a range of datasets and achieves excellent performance even in challenging scenarios, e.g.completion of periodic structures. Our model generalizes surprisingly well to resolutions that are higher than thoseseen at train time, and achieves this at lower parameter & compute costs than the competitive baselines.

## Table of Contents
- [Installation](#installation)
- [LaMa Demos](#paint-by-example-demos)
  - [Diffuser Demo](#paintbyexample-diffuser-demos)
  - [PaintByExample with SAM](#paintbyexample-with-sam)


## TODO
- [x] LaMa Demo with lama-cleaner
- [x] LaMa with SAM
- [ ] LaMa with GroundingDINO
- [ ] LaMa with Grounded-SAM


## Installation
We're using lama-cleaner for this demo, install it as follows:
```bash
pip install lama-cleaner
```
Please refer to [lama-cleaner](https://github.com/Sanster/lama-cleaner) for more details. 

Then install Grounded-SAM follows [Grounded-SAM Installation](https://github.com/IDEA-Research/Grounded-Segment-Anything#installation) for some extension demos.

## LaMa Demos
Here we provide the demos for `LaMa`

### LaMa Demo with lama-cleaner

```bash
cd playground/LaMa
python lama_inpaint_demo.py
```
with the highly organized code lama-cleaner, this demo can be done in about 20 lines of code. The result will be saved as `lama_inpaint_demo.jpg`:

<div align="center">

| Input Image | Mask | Inpaint Output |
|:----:|:----:|:----:|
| ![](https://github.com/IDEA-Research/detrex-storage/blob/main/assets/grounded_sam/lama/example.jpg?raw=true) | ![](https://github.com/IDEA-Research/detrex-storage/blob/main/assets/grounded_sam/lama/mask.png?raw=true) | ![](https://github.com/IDEA-Research/detrex-storage/blob/main/assets/grounded_sam/lama/lama_inpaint_demo.jpg?raw=true) |

</div>

### LaMa with SAM

```bash
cd playground/LaMa
python sam_lama.py
```

**Tips** 
To make it better for inpaint, we should **dilate the mask first** to make it a bit larger to cover the whole region (Thanks a lot for [Inpaint-Anything](https://github.com/geekyutao/Inpaint-Anything) and [Tao Yu](https://github.com/geekyutao) for this)


The `original mask` and `dilated mask` are shown as follows:

<div align="center">

| Mask | Dilated Mask |
|:---:|:---:|
| ![](https://github.com/IDEA-Research/detrex-storage/blob/main/assets/grounded_sam/lama/sam_demo_mask.png?raw=true) | ![](https://github.com/IDEA-Research/detrex-storage/blob/main/assets/grounded_sam/lama/dilated_mask.png?raw=true) |

</div>


And the inpaint result will be saved as `sam_lama_demo.jpg`:

| Input Image | SAM Output | Dilated Mask | LaMa Inpaint |
|:---:|:---:|:---:|:---:|
| ![](https://github.com/IDEA-Research/detrex-storage/blob/main/assets/grounded_sam/paint_by_example/input_image.png?raw=true) | ![](https://github.com/IDEA-Research/detrex-storage/blob/main/assets/grounded_sam/paint_by_example/demo_with_point_prompt.png?raw=true) | ![](https://github.com/IDEA-Research/detrex-storage/blob/main/assets/grounded_sam/lama/dilated_mask.png?raw=true) | ![](https://github.com/IDEA-Research/detrex-storage/blob/main/assets/grounded_sam/lama/sam_lama_demo.jpg?raw=true) |


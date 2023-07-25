## Efficient Grounded-SAM

We're going to combine [Grounding-DINO](https://github.com/IDEA-Research/GroundingDINO) with efficient SAM variants for faster annotating.

<!-- Combining [Grounding-DINO](https://github.com/IDEA-Research/GroundingDINO) and [Fast-SAM](https://github.com/CASIA-IVA-Lab/FastSAM) for faster zero-shot detect and segment anything. -->


### Table of Contents
- [Installation](#installation)
- [Efficient SAM Series](#efficient-sams)
- [Run Grounded-FastSAM Demo](#run-grounded-fastsam-demo)
- [Run Grounded-MobileSAM Demo](#run-grounded-mobilesam-demo)
- [Run Grounded-LightHQSAM Demo](#run-grounded-light-hqsam-demo)


### Installation

- Install [Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything#installation)

- Install [Fast-SAM](https://github.com/CASIA-IVA-Lab/FastSAM#installation)


### Efficient SAMs
Here's the list of Efficient SAM variants:

<div align="center">

| Title | Intro | Description | Links |
|:----:|:----:|:----:|:----:|
| [FastSAM](https://arxiv.org/pdf/2306.12156.pdf) | ![](https://github.com/CASIA-IVA-Lab/FastSAM/blob/main/assets/Overview.png) | The Fast Segment Anything Model(FastSAM) is a CNN Segment Anything Model trained by only 2% of the SA-1B dataset published by SAM authors. The FastSAM achieve a comparable performance with the SAM method at 50× higher run-time speed. | [[Github](https://github.com/CASIA-IVA-Lab/FastSAM)]  [[Demo](https://huggingface.co/spaces/An-619/FastSAM)] |
| [MobileSAM](https://arxiv.org/pdf/2306.14289.pdf) | ![](https://github.com/ChaoningZhang/MobileSAM/blob/master/assets/model_diagram.jpg?raw=true) | MobileSAM performs on par with the original SAM (at least visually) and keeps exactly the same pipeline as the original SAM except for a change on the image encoder. Specifically, we replace the original heavyweight ViT-H encoder (632M) with a much smaller Tiny-ViT (5M). On a single GPU, MobileSAM runs around 12ms per image: 8ms on the image encoder and 4ms on the mask decoder. | [[Github](https://github.com/ChaoningZhang/MobileSAM)] |
| [Light-HQSAM](https://arxiv.org/pdf/2306.01567.pdf) | ![](https://github.com/SysCV/sam-hq/blob/main/figs/sam-hf-framework.png?raw=true) | Light HQ-SAM is based on the tiny vit image encoder provided by MobileSAM. We design a learnable High-Quality Output Token, which is injected into SAM's mask decoder and is responsible for predicting the high-quality mask. Instead of only applying it on mask-decoder features, we first fuse them with ViT features for improved mask details. Refer to [Light HQ-SAM vs. MobileSAM](https://github.com/SysCV/sam-hq#light-hq-sam-vs-mobilesam-on-coco) for more details. | [[Github](https://github.com/SysCV/sam-hq)] |

</div>


### Run Grounded-FastSAM Demo

- Firstly, download the pretrained Fast-SAM weight [here](https://github.com/CASIA-IVA-Lab/FastSAM#model-checkpoints)

- Run the demo with the following script:

```bash
cd Grounded-Segment-Anything

python EfficientSAM/grounded_fast_sam.py --model_path "./FastSAM-x.pt" --img_path "assets/demo4.jpg" --text "the black dog." --output "./output/"
```

- And the results will be saved in `./output/` as:

<div style="text-align: center">

| Input | Text | Output |
|:---:|:---:|:---:|
|![](/assets/demo4.jpg) | "The black dog." | ![](https://github.com/IDEA-Research/detrex-storage/blob/main/assets/grounded_sam/fast_sam/demo4_0_caption_the%20black%20dog.jpg?raw=true) |

</div>


**Note**: Due to the post process of FastSAM, only one box can be annotated at a time, if there're multiple box prompts, we simply save multiple annotate images to `./output` now, which will be modified in the future release.


### Run Grounded-MobileSAM Demo

- Firstly, download the pretrained MobileSAM weight [here](https://github.com/ChaoningZhang/MobileSAM/tree/master/weights)

- Run the demo with the following script:

```bash
cd Grounded-Segment-Anything

python EfficientSAM/grounded_mobile_sam.py
```

- And the result will be saved as `./gronded_mobile_sam_anontated_image.jpg` as:

<div style="text-align: center">

| Input | Text | Output |
|:---:|:---:|:---:|
|![](/assets/demo2.jpg) | "The running dog" | ![](https://github.com/IDEA-Research/detrex-storage/blob/main/assets/grounded_sam/mobile_sam/grounded_mobile_sam_annotated_image.jpg?raw=true) |

</div>


### Run Grounded-Light-HQSAM Demo

- Firstly, download the pretrained Light-HQSAM weight [here](https://github.com/SysCV/sam-hq#model-checkpoints)

- Run the demo with the following script:

```bash
cd Grounded-Segment-Anything

python EfficientSAM/grounded_light_hqsam.py
```

- And the result will be saved as `./gronded_light_hqsam_anontated_image.jpg` as:

<div style="text-align: center">

| Input | Text | Output |
|:---:|:---:|:---:|
|![](/EfficientSAM/LightHQSAM/example_light_hqsam.png) | "Bench" | ![](/EfficientSAM/LightHQSAM/grounded_light_hqsam_annotated_image.jpg) |

</div>
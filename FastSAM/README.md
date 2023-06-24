## Grounded-FastSAM

Combining [Grounding-DINO](https://github.com/IDEA-Research/GroundingDINO) and [Fast-SAM](https://github.com/CASIA-IVA-Lab/FastSAM) for faster zero-shot detect and segment anything.


### Contents
- [Installation](#installation)
- [Run Grounded-Fast-SAM Demo](#run-grounded-fast-sam-demo)

### Catelog
- [x] Support Grounded-FastSAM
- [ ] Support better visualization for Grounded-FastSAM
- [ ] Support RAM-Grounded-FastSAM for faster automatic labelling

### Installation

- Install [Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything#installation)

- Install [Fast-SAM](https://github.com/CASIA-IVA-Lab/FastSAM#installation)

### Run Grounded-Fast-SAM Demo

- Firstly, download the pretrained Fast-SAM weight [here](https://github.com/CASIA-IVA-Lab/FastSAM#model-checkpoints)

- Run the demo with the following script:

```bash
cd Grounded-Segment-Anything

python FastSAM/grounded_fast_sam.py --model_path "./FastSAM-x.pt" --img_path "assets/demo4.jpg" --text "the black dog." --output "./output/"
```

- And the results will be saved in `./output/` as:

<div style="text-align: center">

| Input | Text | Output |
|:---:|:---:|:---:|
|![](/assets/demo4.jpg) | "The black dog." | ![](https://github.com/IDEA-Research/detrex-storage/blob/main/assets/grounded_sam/fast_sam/demo4_0_caption_the%20black%20dog.jpg?raw=true) |

</div>


**Note**: Due to the post process of FastSAM, only one box can be annotated at a time, if there're multiple box prompts, we simply save multiple annotate images to `./output` now, which will be modified in the future release.


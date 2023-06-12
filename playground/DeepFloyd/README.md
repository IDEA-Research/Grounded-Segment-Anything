## DeepFloyd

:grapes: [[Official Project Page](https://github.com/deep-floyd/IF)] &nbsp; :apple:[[Official Online Demo](https://huggingface.co/spaces/DeepFloyd/IF)]

> DeepFloyd IF is a novel state-of-the-art open-source text-to-image model with a high degree of photorealism and language understanding. 

We've thoughtfully put together some important details for you to keep in mind while using the DeepFloyd models. We sincerely hope this will assist you in creating even more interesting demos with IF. Enjoy your creative journey!

## Table of Contents
- [Installation Details](#installation)
  - [Detailed installation guide](#detailed-installation-guide)
  - [Additional note for bug fixing](#additional-notes-for-bug-fixing)
- [Requirements before running demo](#requirements-before-running-demos)
- [DeepFloyd Demos](#deepfloyd-demos)
  - [Dream: Text to Image](#dream)
  - [Style Transfer](#style-transfer)


## TODO
- [x] Add installation guide (Continual Updating)
- [x] Test Text-to-Image model
- [x] Test Style-Transfer model
- [ ] Add Inpaint demo (seems not work well)
- [ ] Add SAM inpaint and Grounded-SAM inpaint demo

## Installation
### Detailed installation guide
There're more things you should take care for installing DeepFloyd despite of their official guide. You can install DeepFloyd as follows:

- Create a new environment using `Python=3.10`

```bash
conda create -n floyd python=3.10 -y
conda activate floyd
```

- DeepFloyd need [xformers](https://github.com/facebookresearch/xformers) to accelerate some attention mechanism and reduce the GPU memory usage. And `xformers` requires at least [PyTorch 1.12.1, PyTorch 1.13.1 or 2.0.0 installed with conda](https://pytorch.org/get-started/locally/).
  - If you only have CUDA 11.4 or lower CUDA version installed, you can only PyTorch 1.12.1 locally as:
  ```bash
  conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -c pytorch
  ```
  - After installing PyTorch, it's highly recommended to install xformers using conda:
  ```bash
  conda install xformers -c xformers
  ```

- Then install deepfloyd following their official guidance:
```bash
pip install deepfloyd_if==1.0.2rc0
pip install git+https://github.com/openai/CLIP.git --no-deps
```

### Additional notes for bug fixing

- [Attention] To use DeepFloyd with diffusers for saving GPU memory usage, you should update your transformers to at least `4.27.0` and accelerate to `0.17.0`.

```bash
pip install transformers==4.27.1
pip install accelerate==0.17.0
```

- And refer to [DeepFloyd/issue64](https://github.com/deep-floyd/IF/pull/64), there are some bugs with inpainting demos, you need `protobuf==3.19.0` to load T5Embedder and `scikit-image` for inpainting

```bash
pip install protobuf==3.19.0
pip install scikit-image
```

However this bug has not been updated to the python package of `DeepFloyd`, so the users should update the code manually follow issue64 or install `DeepFloyd` locally as:

```bash
git clone https://github.com/deep-floyd/IF.git
cd IF
pip install -e .
```

## Requirements before running demos
Before running DeepFloyd demo, please refer to [Integration with DIffusers](https://github.com/deep-floyd/IF#integration-with--diffusers) for some requirements for the pretrained weights.

If you want to download the weights into **specific dir**, you can set `cache_dir` as follows:

- *Under diffusers*
```python
from diffusers import DiffusionPipeline
from diffusers.utils import pt_to_pil
import torch

cache_dir = "path/to/specific_dir"
# stage 1
stage_1 = DiffusionPipeline.from_pretrained(
    "DeepFloyd/IF-I-XL-v1.0", 
    variant="fp16", 
    torch_dtype=torch.float16,
    cache_dir=cache_dir  # loading model from specific dir
  )
stage_1.enable_xformers_memory_efficient_attention()  # remove line if torch.__version__ >= 2.0.0
stage_1.enable_model_cpu_offload()
```

- *Runing locally*
```python
from deepfloyd_if.modules import IFStageI, IFStageII, StableStageIII
from deepfloyd_if.modules.t5 import T5Embedder

cache_dir = "path/to/cache_dir"
device = 'cuda:0'
if_I = IFStageI('IF-I-XL-v1.0', device=device, cache_dir=cache_dir)
if_II = IFStageII('IF-II-L-v1.0', device=device, cache_dir=cache_dir)
if_III = StableStageIII('stable-diffusion-x4-upscaler', device=device, cache_dir=cache_dir)
t5 = T5Embedder(device="cpu", cache_dir=cache_dir)
```

## DeepFloyd Demos

- 16GB vRAM for IF-I-XL (4.3B text to 64x64 base module) & IF-II-L (1.2B to 256x256 upscaler module)
- 24GB vRAM for IF-I-XL (4.3B text to 64x64 base module) & IF-II-L (1.2B to 256x256 upscaler module) & Stable x4 (to 1024x1024 upscaler)
- ***(Highlight)*** `xformers` and set env variable `FORCE_MEM_EFFICIENT_ATTN=1`, which may help you to save lots of GPU memory usage
```bash
export FORCE_MEM_EFFICIENT_ATTN=1
```

### Dream
The `text-to-image` mode for DeepFloyd
```python
cd playground/DeepFloyd

export FORCE_MEM_EFFICIENT_ATTN=1 
python dream.py
```
It takes around `26GB` GPU memory usage for this demo. You can download the following awesome generated images from [inpaint playground storage](https://github.com/IDEA-Research/detrex-storage/tree/main/assets/grounded_sam/inpaint_playground).

<!-- <div style="text-align: center;">
    <img src="./example/dream1.jpg" style="margin:auto;" width="60%">
</div> -->


| Prompt (Generated by GPT-4) | Generated Image |
|:----   |  :----: |
| Underneath the galaxy sky, luminescent stars scatter across the vast expanse like diamond dust. Swirls of cosmic purple and blue nebulae coalesce, creating an ethereal canvas. A solitary tree silhouetted against the astral backdrop, roots burrowed deep into the earth, reaching towards the heavens. Leaves shimmer, reflecting the stellar light show. A lone figure, small against the celestial spectacle, contemplates their insignificance in the grandeur of the universe. The galaxy's reflection on a still, tranquil lake creates a stunning mirror image, tying the earth and cosmos together in a mesmerizing dance of light, space, and time. | ![](https://github.com/IDEA-Research/detrex-storage/blob/main/assets/grounded_sam/inpaint_playground/dream1.jpg?raw=True) |
|Beneath the vast sky, a mesmerizing seascape unfolds. The cerulean sea stretches out to infinity, its surface gently disturbed by the breath of the wind, creating delicate ripples. Sunlight dances on the water, transforming the ocean into a shimmering tapestry of light and shadow. A solitary sailboat navigates the expanse, its white sail billowing against the sapphire backdrop. Nearby, a lighthouse stands resolute on a rocky outcrop, its beacon piercing through the soft maritime mist. Shoreline meets the sea in a frothy embrace, while seagulls wheel overhead, their cries echoing the eternal song of the sea. The scent of salt and freedom fills the air, painting a picture of unbound exploration and serene beauty. | ![](https://github.com/IDEA-Research/detrex-storage/blob/main/assets/grounded_sam/inpaint_playground/dream2.jpg?raw=True) |
| In the heart of the wilderness, an enchanting forest reveals itself. Towering trees, their trunks sturdy and thick, reach skyward, their leafy canopies forming a natural cathedral. Verdant moss clings to bark, and tendrils of ivy climb ambitiously towards the sun-dappled treetops. The forest floor is a tapestry of fallen leaves, sprinkled with delicate wildflowers. The soft chatter of wildlife resonates, while a nearby brook babbles, its clear waters winking in the dappled light. Sunrays filter through the foliage, casting an emerald glow that dances on the woodland floor. Amidst the tranquility, the forest teems with life, whispering ancient secrets on the breeze. |![](https://github.com/IDEA-Research/detrex-storage/blob/main/assets/grounded_sam/inpaint_playground/dream3.jpg?raw=True) |


### Style Transfer
Download the original image from [here](https://github.com/IDEA-Research/detrex-storage/blob/main/assets/grounded_sam/inpaint_playground/style_transfer/original.jpg), which is borrowed from DeepFloyd official image.

<div style="text-align: center">
<img src="https://github.com/IDEA-Research/detrex-storage/blob/main/assets/grounded_sam/inpaint_playground/style_transfer/original.jpg?raw=True" width=50%>
</div>

```python
cd playground/DeepFloyd

export FORCE_MEM_EFFICIENT_ATTN=1 
python style_transfer.py
```

Style | Transfer Image (W/O SuperResolution) |
|  :----: |  :----: |
| *colorful and cute kawaii art* | ![](https://github.com/IDEA-Research/detrex-storage/blob/main/assets/grounded_sam/inpaint_playground/style_transfer/style_transfer_0.jpg?raw=True) |
| *boho-chic textile patterns* | ![](https://github.com/IDEA-Research/detrex-storage/blob/main/assets/grounded_sam/inpaint_playground/style_transfer/style_transfer_1.jpg?raw=True) |

from io import BytesIO

import torch

import PIL
import requests
from diffusers import RePaintPipeline, RePaintScheduler


def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")


img_url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/repaint/celeba_hq_256.png"
mask_url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/repaint/mask_256.png"

# Load the original image and the mask as PIL images
original_image = download_image(img_url).resize((256, 256))
mask_image = download_image(mask_url).resize((256, 256))

# Load the RePaint scheduler and pipeline based on a pretrained DDPM model
DEVICE = "cuda:1"
CACHE_DIR = "/comp_robot/rentianhe/weights/diffusers/"
scheduler = RePaintScheduler.from_pretrained("google/ddpm-ema-celebahq-256", cache_dir=CACHE_DIR)
pipe = RePaintPipeline.from_pretrained("google/ddpm-ema-celebahq-256", scheduler=scheduler, cache_dir=CACHE_DIR)
pipe = pipe.to(DEVICE)

generator = torch.Generator(device=DEVICE).manual_seed(0)
output = pipe(
    image=original_image,
    mask_image=mask_image,
    num_inference_steps=250,
    eta=0.0,
    jump_length=10,
    jump_n_sample=10,
    generator=generator,
)
inpainted_image = output.images[0]
inpainted_image.save("./repaint_demo.jpg")
# !pip install diffusers transformers

import requests
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from diffusers import DiffusionPipeline

from segment_anything import sam_model_registry, SamPredictor


"""
Step 1: Download and preprocess example demo images
"""
def download_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")


img_url = "https://github.com/IDEA-Research/detrex-storage/blob/main/assets/grounded_sam/paint_by_example/input_image.png?raw=true"
# example_url = "https://github.com/IDEA-Research/detrex-storage/blob/main/assets/grounded_sam/paint_by_example/pomeranian_example.jpg?raw=True"
# example_url = "https://github.com/IDEA-Research/detrex-storage/blob/main/assets/grounded_sam/paint_by_example/example_image.jpg?raw=true"
example_url = "https://github.com/IDEA-Research/detrex-storage/blob/main/assets/grounded_sam/paint_by_example/labrador_example.jpg?raw=true"

init_image = download_image(img_url).resize((512, 512))
example_image = download_image(example_url).resize((512, 512))


"""
Step 2: Initialize SAM and PaintByExample models
"""

DEVICE = "cuda:1"

# SAM
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = "/comp_robot/rentianhe/code/Grounded-Segment-Anything/sam_vit_h_4b8939.pth"
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
sam_predictor = SamPredictor(sam)
sam_predictor.set_image(np.array(init_image))

# PaintByExample Pipeline
CACHE_DIR = "/comp_robot/rentianhe/weights/diffusers/"
pipe = DiffusionPipeline.from_pretrained(
    "Fantasy-Studio/Paint-by-Example",
    torch_dtype=torch.float16,
    cache_dir=CACHE_DIR,
)
pipe = pipe.to(DEVICE)


"""
Step 3: Get masks with SAM by prompt (box or point) and inpaint the mask region by example image.
"""

input_point = np.array([[350, 256]])
input_label = np.array([1])  # positive label

masks, _, _ = sam_predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=False
)
mask = masks[0]  # [1, 512, 512] to [512, 512] np.ndarray
mask_pil = Image.fromarray(mask)

mask_pil.save("./mask.jpg")

image = pipe(
    image=init_image, 
    mask_image=mask_pil, 
    example_image=example_image, 
    num_inference_steps=500, 
    guidance_scale=9.0
).images[0]

image.save("./paint_by_example_demo.jpg")

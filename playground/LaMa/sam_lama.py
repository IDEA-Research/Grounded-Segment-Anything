# !pip install diffusers transformers

import requests
import cv2
import numpy as np
import PIL
from PIL import Image
from io import BytesIO

from segment_anything import sam_model_registry, SamPredictor

from lama_cleaner.model.lama import LaMa
from lama_cleaner.schema import Config

"""
Step 1: Download and preprocess demo images
"""
def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


img_url = "https://github.com/IDEA-Research/detrex-storage/blob/main/assets/grounded_sam/paint_by_example/input_image.png?raw=true"


init_image = download_image(img_url)
init_image = np.asarray(init_image)


"""
Step 2: Initialize SAM and LaMa models
"""

DEVICE = "cuda:1"

# SAM
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = "/comp_robot/rentianhe/code/Grounded-Segment-Anything/sam_vit_h_4b8939.pth"
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
sam_predictor = SamPredictor(sam)
sam_predictor.set_image(init_image)

# LaMa
model = LaMa(DEVICE)


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
masks = masks.astype(np.uint8) * 255
# mask_pil = Image.fromarray(masks[0])  # simply save the first mask


"""
Step 4: Dilate Mask to make it more suitable for LaMa inpainting

The idea behind dilate mask is to mask a larger region which will be better for inpainting.

Borrowed from Inpaint-Anything: https://github.com/geekyutao/Inpaint-Anything/blob/main/utils/utils.py#L18
"""

def dilate_mask(mask, dilate_factor=15):
    mask = mask.astype(np.uint8)
    mask = cv2.dilate(
        mask,
        np.ones((dilate_factor, dilate_factor), np.uint8),
        iterations=1
    )
    return mask

def save_array_to_img(img_arr, img_p):
    Image.fromarray(img_arr.astype(np.uint8)).save(img_p)

# [1, 512, 512] to [512, 512] and save mask
save_array_to_img(masks[0], "./mask.png")

mask = dilate_mask(masks[0], dilate_factor=15)

save_array_to_img(mask, "./dilated_mask.png")

"""
Step 5: Run LaMa inpaint model
"""
result = model(init_image, mask, Config(hd_strategy="Original", ldm_steps=20, hd_strategy_crop_margin=128, hd_strategy_crop_trigger_size=800, hd_strategy_resize_limit=800))
cv2.imwrite("sam_lama_demo.jpg", result)

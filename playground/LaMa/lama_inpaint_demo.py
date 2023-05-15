import cv2
import PIL
import requests
import numpy as np
from lama_cleaner.model.lama import LaMa
from lama_cleaner.schema import Config


def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


img_url = "https://raw.githubusercontent.com/Sanster/lama-cleaner/main/assets/dog.jpg"
mask_url = "https://user-images.githubusercontent.com/3998421/202105351-9fcc4bf8-129d-461a-8524-92e4caad431f.png"

image = np.asarray(download_image(img_url))
mask = np.asarray(download_image(mask_url).convert("L"))

# set to GPU for faster inference
model = LaMa("cpu")
result = model(image, mask, Config(hd_strategy="Original", ldm_steps=20, hd_strategy_crop_margin=128, hd_strategy_crop_trigger_size=800, hd_strategy_resize_limit=800))
cv2.imwrite("lama_inpaint_demo.jpg", result)
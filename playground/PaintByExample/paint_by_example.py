# !pip install diffusers transformers

import PIL
import requests
import torch
from io import BytesIO
from diffusers import DiffusionPipeline


"""
Step 1: Download demo images
"""
def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")


img_url = "https://github.com/IDEA-Research/detrex-storage/blob/main/assets/grounded_sam/paint_by_example/input_image.png?raw=true"
mask_url = "https://github.com/IDEA-Research/detrex-storage/blob/main/assets/grounded_sam/paint_by_example/mask.png?raw=true"
example_url = "https://github.com/IDEA-Research/detrex-storage/blob/main/assets/grounded_sam/paint_by_example/pomeranian_example.jpg?raw=True"
# example_url = "https://raw.githubusercontent.com/Fantasy-Studio/Paint-by-Example/main/examples/reference/example_1.jpg"

init_image = download_image(img_url).resize((512, 512))
mask_image = download_image(mask_url).resize((512, 512))
example_image = download_image(example_url).resize((512, 512))


"""
Step 2: Download pretrained weights and initialize model
"""
# set cache dir to store the weights
cache_dir = "/comp_robot/rentianhe/weights/diffusers/"

pipe = DiffusionPipeline.from_pretrained(
    "Fantasy-Studio/Paint-by-Example",
    torch_dtype=torch.float16,
    cache_dir=cache_dir,
)
# set to device
pipe = pipe.to("cuda:1")


"""
Step 3: Run PaintByExample pipeline and save image
"""
image = pipe(
    image=init_image, 
    mask_image=mask_image, 
    example_image=example_image,
    num_inference_steps=200,
).images[0]

image.save("./paint_by_example_demo.jpg")

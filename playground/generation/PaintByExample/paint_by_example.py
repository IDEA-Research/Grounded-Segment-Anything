# !pip install diffusers transformers

import PIL
import requests
import torch
from io import BytesIO
from diffusers import DiffusionPipeline


def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")


img_url = "https://raw.githubusercontent.com/Fantasy-Studio/Paint-by-Example/main/examples/image/example_1.png"
mask_url = "https://raw.githubusercontent.com/Fantasy-Studio/Paint-by-Example/main/examples/mask/example_1.png"
example_url = "https://raw.githubusercontent.com/Fantasy-Studio/Paint-by-Example/main/examples/reference/example_1.jpg"

init_image = download_image(img_url).resize((512, 512))
mask_image = download_image(mask_url).resize((512, 512))
example_image = download_image(example_url).resize((512, 512))

# set cache dir to store the weights
cache_dir = "/comp_robot/rentianhe/weights/diffusers/"

pipe = DiffusionPipeline.from_pretrained(
    "Fantasy-Studio/Paint-by-Example",
    torch_dtype=torch.float16,
    cache_dir=cache_dir,
)
pipe = pipe.to("cuda:1")

image = pipe(image=init_image, mask_image=mask_image, example_image=example_image).images[0]

image.save("./paint_by_example_demo.jpg")

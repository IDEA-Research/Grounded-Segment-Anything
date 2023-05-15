import PIL
import requests
from io import BytesIO
from torchvision.transforms import ToTensor

from deepfloyd_if.modules import IFStageI, IFStageII, StableStageIII
from deepfloyd_if.modules.t5 import T5Embedder
from deepfloyd_if.pipelines import inpainting

def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")

img_url = "https://raw.githubusercontent.com/Fantasy-Studio/Paint-by-Example/main/examples/image/example_1.png"
mask_url = "https://raw.githubusercontent.com/Fantasy-Studio/Paint-by-Example/main/examples/mask/example_1.png"

init_image = download_image(img_url).resize((512, 512))
mask_image = download_image(mask_url).resize((512, 512))

# convert mask_image to torch.Tensor to avoid bug
mask_image = ToTensor()(mask_image).unsqueeze(0)  # (1, 3, 512, 512)

# Run locally
device = 'cuda:5'
cache_dir = "/comp_robot/rentianhe/weights/IF/"
if_I = IFStageI('IF-I-L-v1.0', device=device, cache_dir=cache_dir)
if_II = IFStageII('IF-II-L-v1.0', device=device, cache_dir=cache_dir)
if_III = StableStageIII('stable-diffusion-x4-upscaler', device=device, cache_dir=cache_dir)
t5 = T5Embedder(device=device, cache_dir=cache_dir)
result = inpainting(
    t5=t5, if_I=if_I,
    if_II=if_II,
    if_III=if_III,
    support_pil_img=init_image,
    inpainting_mask=mask_image,
    prompt=[
        'A Panda'
    ],
    seed=42,
    if_I_kwargs={
        "guidance_scale": 7.0,
        "sample_timestep_respacing": "10,10,10,10,10,0,0,0,0,0",
        'support_noise_less_qsample_steps': 0,
    },
    if_II_kwargs={
        "guidance_scale": 4.0,
        'aug_level': 0.0,
        "sample_timestep_respacing": '100',
    },
    if_III_kwargs={
        "guidance_scale": 9.0,
        "noise_level": 20,
        "sample_timestep_respacing": "75",
    },
)
if_I.show(result['I'], 2, 3)
if_I.show(result['II'], 2, 6)
if_I.show(result['III'], 2, 14)


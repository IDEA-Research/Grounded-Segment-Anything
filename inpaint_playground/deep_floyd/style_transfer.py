from PIL import Image

from deepfloyd_if.modules import IFStageI, IFStageII, StableStageIII
from deepfloyd_if.modules.t5 import T5Embedder
from deepfloyd_if.pipelines import style_transfer

# Run locally
device = 'cuda:5'
cache_dir = "/comp_robot/rentianhe/weights/IF"
if_I = IFStageI('IF-I-XL-v1.0', device=device, cache_dir=cache_dir)
if_II = IFStageII('IF-II-L-v1.0', device=device, cache_dir=cache_dir)
t5 = T5Embedder(device=device, cache_dir=cache_dir)

# Style generate from GPT-4
style_prompt = [
    "in style of colorful and cute kawaii art",
    "in style of boho-chic textile patterns",
]

raw_pil_image = Image.open("/path/to/image")

result = style_transfer(
    t5=t5, if_I=if_I, if_II=if_II,
    support_pil_img=raw_pil_image,
    style_prompt=style_prompt,
    seed=42,
    if_I_kwargs={
        "guidance_scale": 10.0,
        "sample_timestep_respacing": "10,10,10,10,10,10,10,10,0,0",
        'support_noise_less_qsample_steps': 5,
    },
    if_II_kwargs={
        "guidance_scale": 4.0,
        "sample_timestep_respacing": 'smart50',
        "support_noise_less_qsample_steps": 5,
    },
)
import pdb; pdb.set_trace()
result['III'][0].save("./dream_figure.jpg")


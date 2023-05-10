from deepfloyd_if.modules import IFStageI, IFStageII, StableStageIII
from deepfloyd_if.modules.t5 import T5Embedder
from deepfloyd_if.pipelines import dream

# from huggingface_hub import login
# login()

# Run locally
device = 'cuda:5'
cache_dir = "/comp_robot/rentianhe/weights/IF"
if_I = IFStageI('IF-I-L-v1.0', device=device, cache_dir=cache_dir)
if_II = IFStageII('IF-II-L-v1.0', device=device, cache_dir=cache_dir)
if_III = StableStageIII('stable-diffusion-x4-upscaler', device=device, cache_dir=cache_dir)
t5 = T5Embedder(device=device, cache_dir=cache_dir)

prompt = "Beneath the vast sky, a mesmerizing seascape unfolds. The cerulean sea stretches out to infinity, its surface gently disturbed by the breath of the wind, creating delicate ripples. Sunlight dances on the water, transforming the ocean into a shimmering tapestry of light and shadow. A solitary sailboat navigates the expanse, its white sail billowing against the sapphire backdrop. Nearby, a lighthouse stands resolute on a rocky outcrop, its beacon piercing through the soft maritime mist. Shoreline meets the sea in a frothy embrace, while seagulls wheel overhead, their cries echoing the eternal song of the sea. The scent of salt and freedom fills the air, painting a picture of unbound exploration and serene beauty."
count = 1

result = dream(
    t5=t5, if_I=if_I, if_II=if_II, if_III=if_III,
    prompt=[prompt]*count,
    seed=42,
    if_I_kwargs={
        "guidance_scale": 7.0,
        "sample_timestep_respacing": "smart100",
    },
    if_II_kwargs={
        "guidance_scale": 4.0,
        "sample_timestep_respacing": "smart50",
    },
    if_III_kwargs={
        "guidance_scale": 9.0,
        "noise_level": 20,
        "sample_timestep_respacing": "75",
    },
)
result['III'][0].save("./dream_figure.jpg")


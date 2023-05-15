from deepfloyd_if.modules import IFStageI, IFStageII, StableStageIII
from deepfloyd_if.modules.t5 import T5Embedder
from deepfloyd_if.pipelines import dream

# Run locally
device = 'cuda'
cache_dir = "/path/to/storage/IF"
if_I = IFStageI('IF-I-L-v1.0', device=device, cache_dir=cache_dir)
if_II = IFStageII('IF-II-L-v1.0', device=device, cache_dir=cache_dir)
if_III = StableStageIII('stable-diffusion-x4-upscaler', device=device, cache_dir=cache_dir)
t5 = T5Embedder(device=device, cache_dir=cache_dir)

prompt = "In the heart of the wilderness, an enchanting forest reveals itself. \
    Towering trees, their trunks sturdy and thick, reach skyward, their leafy canopies \
    forming a natural cathedral. Verdant moss clings to bark, and tendrils of ivy climb ambitiously towards the sun-dappled treetops. \
    The forest floor is a tapestry of fallen leaves, sprinkled with delicate wildflowers. The soft chatter of wildlife resonates, while a nearby brook babbles, its clear waters winking in the dappled light. \
    Sunrays filter through the foliage, casting an emerald glow that dances on the woodland floor. Amidst the tranquility, the forest teems with life, whispering ancient secrets on the breeze."
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


import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained("/project/infattllm/lyangbk/huggingface/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

seed = 666
sd_generator = torch.manual_seed(seed)

image = pipe(
    prompt="Anime style illustration of a girl wearing a suit.",
    num_inference_steps=28,
    height=1024,
    width=1024,
    generator=sd_generator,
    guidance_scale=1.0,
).images[0]

image.save("diffusers_sd3.png")

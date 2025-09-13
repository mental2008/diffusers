import time

import torch

from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained("/project/infattllm/lyangbk/huggingface/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

seed = 666
sd_generator = torch.manual_seed(seed)

# warm up
pipe(
    prompt=["Anime style illustration of a girl wearing a suit.", "Anime style illustration of a girl wearing a suit."],
    negative_prompt=["NSFW, nude, naked, porn, ugly", "NSFW, nude, naked, porn, ugly"],
    num_inference_steps=28,
    height=1024,
    width=1024,
    generator=sd_generator,
    guidance_scale=7.0,
).images[0]

start_time = time.time()
image = pipe(
    prompt="Anime style illustration of a girl wearing a suit.",
    negative_prompt="NSFW, nude, naked, porn, ugly",
    num_inference_steps=28,
    height=1024,
    width=1024,
    generator=sd_generator,
    guidance_scale=7.0,
).images[0]
end_time = time.time()
print(f"Time taken to execute model: {end_time - start_time} seconds")

image.save("diffusers_sd3_cfg.png")

import time

import torch
from diffusers import StableDiffusion3ControlNetPipeline
from diffusers.models import SD3ControlNetModel, SD3MultiControlNetModel
from diffusers.utils import load_image


controlnet = SD3ControlNetModel.from_pretrained("/project/infattllm/lyangbk/huggingface/sd3-controlnet-canny", torch_dtype=torch.float16)
controlnet = controlnet.to("cuda")

pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
    "/project/infattllm/lyangbk/huggingface/stable-diffusion-3-medium-diffusers", 
    controlnet=controlnet, 
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

seed = 666
sd_generator = torch.manual_seed(seed)

control_image = load_image("imgs/canny.jpg")
prompt = 'Anime style illustration of a girl wearing a suit.'
negative_prompt = None

# warm up
pipe(
    prompt, 
    negative_prompt=negative_prompt,
    control_image=control_image, 
    height=1024, 
    width=1024, 
    num_inference_steps=28,
    generator=sd_generator,
    controlnet_conditioning_scale=0.7,
    guidance_scale=1.0,
).images[0]

start_time = time.time()
image = pipe(
    prompt, 
    negative_prompt=negative_prompt,
    control_image=control_image, 
    height=1024, 
    width=1024, 
    num_inference_steps=28,
    generator=sd_generator,
    controlnet_conditioning_scale=0.7,
    guidance_scale=1.0,
).images[0]
end_time = time.time()
print(f"Time taken to execute model: {end_time - start_time} seconds")

image.save("diffusers_sd3_controlnet.png")

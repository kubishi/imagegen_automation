import torch
from diffusers import StableDiffusionPipeline
from torch.cuda import empty_cache
import pathlib
from uuid import uuid4
from huggingface_hub import snapshot_download

thisdir = pathlib.Path(__file__).parent.absolute()

cache_dir = pathlib.Path("/home/c_kpathak1@lmumain.edu/.cache/huggingface/hub")

pipe = StableDiffusionPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V5.1_noVAE", 
    torch_dtype=torch.bfloat16, 
    use_safetensors=True,
    cache_dir=cache_dir
).to("cuda")

negative_prompt = (
    "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), "
    "text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers,"
    "mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated,"
    "bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms," 
    "missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
    )
# Image generation function
def generate_image(prompt: str, save_path: pathlib.Path):
    image = pipe(
        prompt,
        height=1024,
        width=1024,
        guidance_scale=5.5, # guidance scale controls how strongly the image follows the text prompt
        num_inference_steps=30, # more steps -> higher quality, slower generation
        generator=torch.Generator("cuda").manual_seed(0),
        negative_prompt=negative_prompt,
        num_images_per_prompt=1,
    ).images[0]

    save_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(save_path)

    # Clear GPU memory after each prompt
    empty_cache()

def main():
    savedir = thisdir / "output"
    savedir.mkdir(exist_ok=True)
    while True:
        generate_image(
            prompt=input("Enter a prompt: "),
            save_path=savedir / f"{uuid4()}.png"
        )

if __name__ == "__main__":
    main()

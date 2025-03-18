import torch
from diffusers import StableDiffusionXLPipeline
from torch.cuda import empty_cache
import pathlib
from uuid import uuid4
from huggingface_hub import snapshot_download

thisdir = pathlib.Path(__file__).parent.absolute()

cache_dir = pathlib.Path("/home/c_kpathak1@lmumain.edu/.cache/huggingface/hub")
# loraspath = pathlib.Path("/data/share/models/sdxl-loras")
# lora_path = loraspath / "lcm-lora-sdxl"

# if not lora_path.exists():
#     snapshot_download(repo_id="latent-consistency/lcm-lora-sdxl", local_dir=str(lora_path))

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", 
    torch_dtype=torch.bfloat16, 
    use_safetensors=True,
    cache_dir=cache_dir
).to("cuda")

# pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

# # Load and fuse the LCM-LoRA weights
# pipe.load_lora_weights(str(lora_path))
# pipe.fuse_lora()


# Image generation function
def generate_image(prompt: str, save_path: pathlib.Path):
    image = pipe(
        prompt,
        height=1024,
        width=1024,
        guidance_scale=8.0, # guidance scale controls how strongly the image follows the text prompt
        num_inference_steps=35, # more steps -> higher quality, slower generation
        generator=torch.Generator("cuda").manual_seed(0),
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

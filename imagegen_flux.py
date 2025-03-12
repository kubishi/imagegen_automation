import torch
from diffusers import FluxPipeline
from torch.cuda import empty_cache
import pathlib
from uuid import uuid4
from huggingface_hub import login

thisdir = pathlib.Path(__file__).parent.absolute()

# log in with hugging face - only need to do this first time
# login()

loraspath = pathlib.Path("/data/share/models/sd-loras")

pipe: FluxPipeline = None
def get_pipe():
    global pipe
    if pipe is None:
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", 
            torch_dtype=torch.bfloat16, 
            device_map="balanced"
        )
        # lora_path = loraspath / "char_portraits_flux_lora_v1"
        # pipe.load_lora_weights(str(lora_path))
        # pipe.fuse_lora()
    return pipe

# Image generation function
def generate_image(prompt: str, save_path: pathlib.Path):
    pipe = get_pipe()    
    image = pipe(
        prompt,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=30,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0),
        num_images_per_prompt=1
    ).images[0]
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

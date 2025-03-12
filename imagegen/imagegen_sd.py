import torch
from diffusers import StableDiffusionPipeline
from torch.cuda import empty_cache
import pathlib
import json
import shutil
import threading
from typing import Any, List, Dict, Union, Callable
from uuid import uuid4
from safetensors.torch import load_file
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Define paths for model storage
this_dir = pathlib.Path(__file__).parent.absolute()
loras_path = pathlib.Path.home() / "share/models/sd-loras"
cache_dir = pathlib.Path.home() / "share/models/huggingface"

# Load Stable Diffusion model pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",  # Change this for different SD versions
    torch_dtype=torch.float16,  # Optimized for GPU
    cache_dir=cache_dir
).to("cuda")

def disable_safety_checker(images, **kwargs):
    return images, [False] * len(images)  

pipe.safety_checker = disable_safety_checker

# Load and apply LoRA weights (optional)
lora_path = loras_path / "my_custom_sd_lora.safetensors"

if lora_path.exists():
    try:
        pipe.load_lora_weights(str(lora_path))  # Load LoRA weights
        pipe.fuse_lora()  # Apply LoRA
        print(f"LoRA weights loaded successfully from: {lora_path}")
    except Exception as e:
        print(f"Error loading LoRA weights: {e}. Proceeding without LoRA.")
else:
    print(f"Warning: LoRA file '{lora_path}' not found! Proceeding without LoRA.")

# Image generation function using Stable Diffusion
def generate_image(prompt: str, save_path: pathlib.Path) -> None:
    """Generates an image based on the given prompt and saves it."""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    image = pipe(
        prompt=prompt,
        negatitive_prompt="A blurry image, low quality, distorted, cropped, or with watermarks",
        height=768,  # Optimal for SD v1.5
        width=768,
        guidance_scale=7.5,  # Higher value = stronger prompt adherence
        num_inference_steps=50,  # More steps = better quality
        generator=torch.Generator("cpu").manual_seed(42),  # Fixed seed for reproducibility
        num_images_per_prompt=1,
    ).images[0]

    image.save(save_path)
    print(f"Image saved: {save_path}")

# Define pipeline for image generation
def pipeline(
        prompts: List[str],
        iterations: int,
        images_per_prompt: int,
        best_n: int,
        num_users: int = 5,
        overwrite: bool = False,
        rating_func: Union[Callable[[str, pathlib.Path], Any], List[Callable[[str, pathlib.Path], Any]]] = None):
    """
    Executes image generation in multiple iterations, saves history, and applies auto-rating if provided.
    """

    history_path = this_dir / "history.json"
    images_path = this_dir / "images_sd"

    if not isinstance(rating_func, list):
        rating_func = [rating_func] * num_users if rating_func else []

    # Reset storage if overwrite is enabled
    if overwrite:
        if images_path.exists():
            shutil.rmtree(images_path)
        if history_path.exists():
            history_path.write_text("[]")

    history: List[Dict[str, Any]] = []
    if history_path.exists():
        history = json.loads(history_path.read_text())

    best_examples = sorted(history, key=lambda x: x.get("rating", 0), reverse=True)[:best_n]

    for prompt in prompts:
        for iteration in range(iterations):
            print(f"Processing Iteration {iteration + 1} for prompt: {prompt}")

            modified_prompts = [prompt] * images_per_prompt  # Placeholder for future modifications

            image_threads = []
            for i, mod_prompt in enumerate(modified_prompts):
                save_path = images_path / f"{prompt.replace(' ', '_')}/iteration_{iteration}/image_{i}.png"
                if save_path.exists():
                    print(f"Skipping existing image: {save_path}")
                    continue

                # Start thread for image generation
                # thread = threading.Thread(target=generate_image, args=(mod_prompt, save_path))
                # thread.start()
                # image_threads.append(thread)
                generate_image(mod_prompt, save_path)  


                history.append({
                    "iteration": iteration,
                    "original_prompt": prompt,
                    "modified_prompt": mod_prompt,
                    "image_path": str(save_path),
                })

                history_path.write_text(json.dumps(history, indent=4, ensure_ascii=False))

            # for thread in image_threads:
            #     thread.join()

            # Apply ratings if enabled
            if rating_func:
                for item in history:
                    if "ratings" not in item:
                        ratings = [rating_func[j](item["modified_prompt"], pathlib.Path(item["image_path"])) for j in range(num_users)]
                        avg_rating = sum(r.rating for r in ratings) / len(ratings)
                        item.update({"ratings": [r.model_dump() for r in ratings], "rating": avg_rating})

                history_path.write_text(json.dumps(history, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    # prompts = ["A hyper-realistic futuristic cityscape", "A medieval castle by a river during sunset"]
    prompt_path = this_dir / "prompts.json"
    prompts: List[str] = json.loads(prompt_path.read_text())

    pipeline(
        prompts=prompts,
        iterations=3,
        images_per_prompt=2,
        best_n=3,
        num_users=3,
        overwrite=True,
        rating_func=None  # Change to auto-rating function if needed
    )

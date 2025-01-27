import random
import pathlib
import json
from typing import List, Dict
import requests
import openai
import os
import dotenv
import shutil
import threading

# Loading environment variables
dotenv.load_dotenv()

# Defining the path to the current directory
thisdir = pathlib.Path(__file__).parent.absolute()
client = openai.Client(api_key=os.getenv("OPEN_API_KEY"))


# Generating images using OpenAI's DALL-E model
def generate_image(prompt: str, save_path: str) -> None:
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )

    image_url = response.data[0].url
    image = requests.get(image_url)

    path = pathlib.Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(image.content)


# Modifying user input prompts using OpenAI's GPT-4 model with the help of function calling OpenAI's API
def modify_prompt(prompt: str, examples: List[Dict[str, str]]) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a prompt modifier that changes user input prompts so "
                "that they are detailed and culturally appropriate, avoiding "
                "stereotypes and assumptions."
            )
        }
    ]
    for example in examples:
        messages.append({
            "role": "user",
            "content": example["original_prompt"]
        })
        messages.append({
            "role": "assistant",
            "content": example["modified_prompt"]
        })
    
    messages.append({
        "role": "user",
        "content": prompt
    })

    # saving the inout and output prompts
    (thisdir / "messages.json").write_text(json.dumps(messages, indent=4))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    return response.choices[0].message.content


# Ranking images randomly
def rate_images(history: List[Dict[str, str]]) -> None:
    for i, item in enumerate(history):
        if not item.get("rating"):
            print(f"Image: {item['image_path']}")
            while True:
                try:
                    rating = input("Rate the image from 1 to 5: ")
                    rating = int(rating)
                    if rating < 1 or rating > 5:
                        raise ValueError("Rating must be between 1 and 5")
                    break
                except ValueError as e:
                    print(e)

            history[i]["rating"] = rating
            # history[i]["rating"] = random.randint(1, 5)


# Sanitizing strings
def sanitize_string(s: str) -> str:
    return s.replace(" ", "_").replace(",", "").replace(":", "").replace(";", "").replace(".", "").replace("?", "").replace("!", "")


# Defining the pipeline
def pipeline(iterations: int,
             images_per_prompt: int,
             best_n: int,
             overwrite: bool = False) -> None:
    prompt_path = thisdir / "prompts.json"
    history_path = thisdir / "history.json"
    images_path = thisdir / "images"

    if overwrite:
        if images_path.exists():
            shutil.rmtree(images_path)
        if history_path.exists():
            history_path.write_text("[]")

    prompts: List[str] = json.loads(prompt_path.read_text())
    history = []

    if history_path.exists():
        history = json.loads(history_path.read_text())

    best_examples = sorted(history, key=lambda x: x["rating"], reverse=True)[:best_n]
    for prompt in prompts:
        for iteration in range(iterations):
            print(f"Processing Iteration {iteration + 1} for prompt: {prompt}")

            modified_prompts = [modify_prompt(prompt, best_examples) for _ in range(images_per_prompt)]

            image_paths = []
            image_threads: List[threading.Thread] = []
            for i, mod_prompt in enumerate(modified_prompts):
                save_path = images_path / f"{sanitize_string(prompt)}/iteration_{iteration}/image_{i}.png"
                if save_path.exists():
                    print(f"Skipping image {save_path}")
                    image_paths.append(str(save_path))
                    continue

                # generate_image(mod_prompt, save_path)
                thread = threading.Thread(target=generate_image, args=(mod_prompt, save_path))
                thread.start()
                image_threads.append(thread)

                history.append({
                    "iteration": iteration,
                    "original_prompt": prompt,
                    "modified_prompt": mod_prompt,
                    "image_path": str(save_path)
                })
                history_path.write_text(json.dumps(history, indent=4))

            for thread in image_threads:
                thread.join()

            rate_images(history)
            history_path.write_text(json.dumps(history, indent=4))
            best_examples = sorted(history, key=lambda x: x["rating"], reverse=True)[:best_n]


def main():
    pipeline(iterations=3, images_per_prompt=5, best_n=3, overwrite=True)

if __name__ == "__main__":
    main()

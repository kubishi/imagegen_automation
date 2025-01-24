import random
import pathlib
import json
from typing import List, Dict
import requests
import openai
import os
import dotenv

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
                "You are a prompt modifier that changes user input prompts so that they are detailed and culturally appropriate."
            )
        },
        *[
            {
                "role": "user",
                "content": f"{example['original_prompt']} -> {example['modified_prompt']}"
            }
            for example in examples
        ],
        {
            "role": "user",
            "content": prompt
        }
    ]

    # saving the inout and output prompts
    (thisdir / "messages.json").write_text(json.dumps(messages, indent=4))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    return response.choices[0].message.content


# Ranking images randomly
def rank_images(image_paths: List[str]) -> int:
    return random.randint(0, len(image_paths) - 1)


# Sanitizing strings
def sanitize_string(s: str) -> str:
    return s.replace(" ", "_").replace(",", "").replace(":", "").replace(";", "").replace(".", "").replace("?", "").replace("!", "")


# Defining the pipeline
def pipeline(iterations: int, images_per_prompt: int) -> None:
    prompts: List[str] = json.loads((thisdir / "prompts.json").read_text())
    history = []

    for prompt in prompts:
        current_prompt = prompt
        for iteration in range(iterations):
            print(f"Processing Iteration {
                  iteration + 1} for prompt: {current_prompt}")

            modified_prompt = [modify_prompt(
                current_prompt, history) for _ in range(images_per_prompt)]

            image_paths = []
            for i, mod_prompt in enumerate(modified_prompt):
                save_path = thisdir / \
                    f"images/{sanitize_string(prompt)
                              }/iteration_{iteration}/image_{i}.png"
                generate_image(mod_prompt, save_path)
                image_paths.append(str(save_path))

                history.append({
                    "original_prompt": current_prompt,
                    "modified_prompt": mod_prompt,
                    "image_path": str(save_path)
                })

            best_image_idx = rank_images(image_paths)
            print(f"Best image selected: {image_paths[best_image_idx]}")

            current_prompt = modified_prompt[best_image_idx]

            (thisdir / "history.json").write_text(json.dumps(history, indent=4))


if __name__ == "__main__":

    json_file_path = "prompts.json"

    iterations = 3
    images_per_prompt = 5

    pipeline(iterations, images_per_prompt)

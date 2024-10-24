import json
import pathlib
from typing import Callable
import openai
import dotenv
import os

import requests

dotenv.load_dotenv()

thisdir = pathlib.Path(__file__).parent.absolute()
client = openai.Client(api_key=os.getenv("OPEN_API_KEY"))


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

def modify_prompt_basic(prompt: str) -> str:
    messages = [
        {
            "role": "user",
            "content": (
                f"Modify the following prompt so that it is detailed and culturally appropriate: {prompt}. "
                "Respond only with the modified prompt."
            )
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    return response.choices[0].message.content

def modify_prompt_few_shot(prompt: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a prompt modifier that changes user input prompts so that they are detailed and culturally appropriate."
            )
        },
        {
            "role": "user",
            "content": (
                "An african boy playing the guitar"
            )
        },
        {
            "role": "assistant",
            "content": (
                "A photorealistic image happy, dark-skinned boy playing the guitar on a stage full of people. "
                "He is wearing a colorful shirt and a pair of jeans. "
                "It is night time but the stage is bright and colorful. "
            )
        },
        {
            "role": "user",
            "content": (
                "A native american boy going to school"
            )
        },
        {
            "role": "assistant",
            "content": (
                "A cartoon image Native American boy, around ten years old, dressed in a comfortable t-shirt and jeans, walks confidently toward his school. "
                "He has a backpack with native american inspired designs slung over one shoulder. "
                "The sun is shining, and he pauses to admire the beauty of the landscape around him, which features rolling hills and vibrant wildflowers. "
                "His school is a modern building in the distance. "
                "Along the way, he greets a few classmates, reflecting the warmth and community spirit of his neighborhood."
            )
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    return response.choices[0].message.content

def sanitize_string(s: str) -> str:
    return s.replace(" ", "_").replace(",", "").replace(":", "").replace(";", "").replace(".", "").replace("?", "").replace("!", "")

def main():
    approaches = {
        "naive": lambda x: x,
        "modified": modify_prompt_basic,
        "few_shot": modify_prompt_few_shot,
    }

    prompts = [
        "A native american girl dancing",
        "An indian man walking in Manhattan",
    ]

    for prompt in prompts:
        for approach, modify_prompt in approaches.items():
            savedir = thisdir / f"images/{sanitize_string(prompt)}_{sanitize_string(approach)}"
            if savedir.exists():
                continue
            modified_prompt = modify_prompt(prompt)
            generate_image(modified_prompt, savedir / "image.png")
            details = {
                "prompt": prompt,
                "modified_prompt": modified_prompt,
                "approach": approach
            }
            (savedir / "details.json").write_text(json.dumps(details, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    main()

import json
import pathlib
from typing import Callable, Dict, List, Tuple, Union
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

def modify_prompt(prompt: str, examples: List[Tuple[str, str]]) -> str:
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
                "content": f"{prompt} -> {modified_prompt}"
            }
            for prompt, modified_prompt in examples
        ],
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
    prompts: List[str] = json.loads((thisdir / "prompts.json").read_text())

    prompts_per_day = 5
    images_per_prompt = 3
    day = 1

    prompts_for_day = prompts[(day - 1) * prompts_per_day:day * prompts_per_day]
    print(prompts_for_day)

    data: List[Dict[str, Union[str, int]]] = json.loads((thisdir / "data.json").read_text())
    num_examples = min(5, ((day-1)*prompts_per_day*images_per_prompt))
    # get top num_examples examples with most votes
    examples = sorted(data, key=lambda x: x["votes"], reverse=True)[:num_examples]

    for prompt in prompts_for_day:
        prompt_num = prompts.index(prompt)
        for i in range(images_per_prompt):
            modified_prompt = modify_prompt(prompt, examples)
            savedir = thisdir / f"images/day_{day}/prompt_{prompt_num}"
            image_path = savedir / f"image_{i}.png"
            if not image_path.exists():
                generate_image(modified_prompt, image_path)
                print(f"Generated image for prompt {prompt_num} - {i}")

                data.append({
                    "prompt": prompt,
                    "modified_prompt": modified_prompt,
                    "image_path": str(image_path),
                    "votes": None
                })

                (thisdir / "data.json").write_text(json.dumps(data, indent=4))
            else:
                print(f"Image already exists for prompt {prompt_num} - {i}")


if __name__ == "__main__":
    main()

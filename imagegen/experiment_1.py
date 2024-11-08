import json
import pathlib
from typing import Callable, List
import openai
import dotenv
import os

import requests

dotenv.load_dotenv()

thisdir = pathlib.Path(__file__).parent.absolute()
client = openai.Client(api_key=os.getenv("OPEN_API_KEY"))

API_TOKEN = "bTXrVANkuOc2N5BgHqxZ3HSAwmZTMiC0XtTWufrB"
DATA_CENTER = "iad1"
BASE_URL = "https://iad1.qualtrics.com/API/v3/surveys"
LIBRARY_ID = "default"


headers = {
    "content-type": "application/json",
    "x-api-token": API_TOKEN,
}

payload = {
    "name": "Image Generation Experiment",
    "isActive": True,
}


response = requests.get(BASE_URL, headers=headers)

if response.status_code == 200:
    print("Access granted. Surveys fetched successfully!")
    print(response.json())
else:
    print(f"Failed to access surveys: {
          response.status_code} - {response.json()}"
          )


# if response.status_code == 200:
#     survey_id = response_data['result']['id']
#     print(f"Survey created with id: {survey_id}")
# else:
#     print(f"Failed to create survey: {response.status_code} - {response_data}")


def upload_image(image_path: str) -> str:
    upload_url = f"{BASE_URL}/libraries/{LIBRARY_ID}/files"

    with open(image_path, 'rb') as file:
        files = {'file': file}
        response = requests.post(upload_url, headers=headers, files=files)

    return response


def get_all_images(directory: str) -> List[str]:
    return [
        os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and (f.endswith('.png'))
    ]


this_dir = f"{thisdir}/images"
image_ids = get_all_images(this_dir)

print(f"Uploading {(image_ids)} images...")


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
                f"Modify the following prompt so that it is detailed and culturally appropriate: {
                    prompt}. "
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
            savedir = thisdir / \
                f"images/{sanitize_string(prompt)}_{sanitize_string(approach)}"
            if savedir.exists():
                continue
            modified_prompt = modify_prompt(prompt)
            generate_image(modified_prompt, savedir / "image.png")
            details = {
                "prompt": prompt,
                "modified_prompt": modified_prompt,
                "approach": approach
            }
            (savedir / "details.json").write_text(json.dumps(details,
                                                             ensure_ascii=False, indent=4))


if __name__ == "__main__":
    main()

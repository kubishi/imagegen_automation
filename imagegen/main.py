import logging
import random
from typing import List, Tuple
from dotenv import load_dotenv  # type: ignore
import os
import json
import openai
import requests
import pathlib
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from io import BytesIO
import torch

load_dotenv()


thisdir = pathlib.Path(__file__).parent.absolute()
client = openai.Client(api_key=os.getenv("OPEN_API_KEY"))

processor = BlipProcessor.from_pretrained(
    "salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "salesforce/blip-image-captioning-base")


def create_prompt_with_rules(prompt: str, rules: dict) -> str:
    return json.dumps({
        "prompt": prompt,
        "rules": rules
    })


def generate_image(prompt: str, save_path: str):
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )

    # download image: image_url = response.data[0].url
    image_url = response.data[0].url
    image = requests.get(image_url)

    path = pathlib.Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(image.content)

    return image_url


def load_image_from_url(image_url: str):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    return img


def generate_prompt(image):
    inputs = processor(image, return_tensors="pt").to("cpu")
    captions_ids = model.generate(**inputs, max_new_tokens=100)
    caption = processor.decode(
        captions_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return caption


def no_prompt_modifier(prompt: str, save_path: str):
    image_url = generate_image(prompt, save_path)

    image = load_image_from_url(image_url)
    description = generate_prompt(image)

    return description


def prompt_modifier(prompt: str, save_path: str):
    messages = [
        {
            "role": "system",
            "content": "Your job is to take user prompts and modify them so that they are culturally sensitive and appropriate and accurate."
        },
        {
            "role": "user",
            "content": "A native american girl dancing"
        },
        {
            "role": "assistant",
            "content": (
                "A native american girl dancing gracefully, with flowing movement and a joyful expression. "
                "She has long, flowing hair that moves with her, and she is wearing a colorful, flowing dress that twirls as she spins. "
                "The background is bright and lively, with soft lighting that highlights her movements. "
                "Her arms are outstretched, and the scene captures the elegance and energy of dance."
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

    modified_prompt = response.choices[0].message.content

    image_url = generate_image(modified_prompt, save_path)

    image = load_image_from_url(image_url)
    description = generate_prompt(image)

    return description


def robot_answerer(prompt: str, question: str, previous_answers: List[Tuple[str, str]]) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a customer asking for an artist to generate an image. "
                f"You have provided them with the prompt: \"{prompt}\". "
                "Now you must answer their question to help them create the image. "
                "Your goal is always to create culturally appropriate images."
            )
        }
    ]

    for q, a in previous_answers:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})

    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    return response.choices[0].message.content


def prompt_clarifier(prompt: str, save_path: str, use_robot_answerer: bool = False):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "ask_question",
                "description": "Ask the user a SHORT question to clarify the prompt.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The question to ask the user to clarify the prompt."
                        }
                    },
                    "required": ["question"],
                    "additionalProperties": False
                }
            }
        }
    ]

    messages = [
        {
            "role": "system",
            "content": (
                "Your job is to take user prompts and clarify them as much as you can so that they are "
                "culturally appropriate and reflect the user's intent. "
                "After the user answers your questions, respond with only the modified prompt."
            )
        },
        {
            'role': 'user',
            'content': prompt
        }
    ]

    responses: List[Tuple[str, str]] = []
    itr = 0
    while True:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools
        )

        messages.append(json.loads(
            response.choices[0].message.model_dump_json()))

        question = ""
        if response.choices[0].message.tool_calls:
            for tool_call in response.choices[0].message.tool_calls:
                if tool_call.function.name == "ask_question":
                    question = json.loads(tool_call.function.arguments)[
                        "question"]
                    if use_robot_answerer:
                        answer = robot_answerer(prompt, question, responses)
                    else:
                        answer = input(f"Answer the question: {question}\n")

                    logging.info(f"Question: {question}, Answer: {answer}")

                    responses.append((question, answer))

                    messages.append({
                        "role": "tool",
                        "content": json.dumps({"answer": answer}),
                        "tool_call_id": tool_call.id
                    })
        else:
            modified_prompt = response.choices[0].message.content

            logging.info(f"Modified Prompt: {modified_prompt}")

            itr += 1
            updated_save_path = f"{save_path}_{itr}.png"

            image_url = generate_image(modified_prompt, updated_save_path)
            image = load_image_from_url(image_url)
            description = generate_prompt(image)

            logging.info("Final Description:" + description)
            return description


def ask_chatGPT(prompt, descriptions):
    chat = f"The original prompt is: {
        prompt}. These are two descriptions of the generated images:\n\n"
    for i, desc in enumerate(descriptions):
        chat += f"Image {i+1}: {desc}\n\n"

    chat += "Which image is more accurate according to the prompt?"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": chat}
        ]
    )

    return response.choices[0].message.content


def get_all_images(directory: str) -> List[str]:
    return [
        os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and (f.endswith('.png'))
    ]


def select_random_images(directory: str, num_images: int = 2) -> List[str]:
    selected_images = get_all_images(directory)
    if len(selected_images) < num_images:
        raise ValueError(
            f"Directory {directory} does not contain enough images")

    return random.sample(selected_images, num_images)


def compare(prompt: str):

    prompt_claifier_description = prompt_clarifier(
        prompt=prompt,
        save_path=f"{thisdir}/images/prompt_clarifier_example",
        use_robot_answerer=False
    )

    prompt_asnwere_description = prompt_clarifier(
        prompt=prompt,
        save_path=f"{thisdir}/images/prompt_answerer_example",
        use_robot_answerer=True
    )

    result = ask_chatGPT(
        prompt=prompt,
        descriptions=[
            prompt_claifier_description,
            prompt_asnwere_description
        ]
    )

    return result


def main():
    logging.basicConfig(level=logging.INFO)
    prompt = "A Native American man going to the office in new york city, manhattan area"

    this_dir = f"{thisdir}/images"

    try:
        img = select_random_images(this_dir)
        print(f"Images: {img}")
    except ValueError as e:
        print(e)

    res = compare(prompt)

    print(f"Result: {res}")


if __name__ == "__main__":
    main()

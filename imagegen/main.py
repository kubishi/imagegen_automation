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
    captions_ids = model.generate(**inputs)
    caption = processor.decode(captions_ids[0], skip_special_tokens=True)
    return caption

def no_prompt_modifier(prompt: str, save_path: str):
    image_url = generate_image(prompt, save_path)

    image = load_image_from_url(image_url)
    description = generate_prompt(image)
    # print(f"Generated Description for no prompt modifier: {description}")

    return description

def rule_based(prompt: str, rules: list, save_path: str):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "choose_rules",
                "description": "Choose rules for prompt which could generate an appropriate image",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "rules": {
                            "type": "array",
                            "description": "List of rules which could make the prompt better in order to get a more accurate image.",
                            "items": {
                                "type": "string"
                            }
                        }
                    },
                    "required": ["rules"],
                    "additionalProperties": False
                }
            }
        }
    ]

    modified_prompt = create_prompt_with_rules(prompt, rules)

    messages = [
        {"role": "system", "content": "Your job is to take user prompts and decide which rules that are more important should be applied to them so that the image generated is appropriate."},
        {"role": "user", "content": modified_prompt}
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools
    )

    rules = []
    for tool_call in response.choices[0].message.tool_calls:
        if tool_call.function.name == "choose_rules":
            rules.extend(json.loads(tool_call.function.arguments)["rules"])

    modified_prompt = f"{prompt}, " + ", ".join(rules)
    # print(f"Prompt: {prompt}")
    # print(f"Rule based Modified Prompt: {modified_prompt}")

    image_url = generate_image(modified_prompt, save_path)

    image = load_image_from_url(image_url)
    description = generate_prompt(image)
    # print(f"Generated Description for rule based modifier: {description}")

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
    # print(f"Prompt: {prompt}")
    # print(f"Prompt based Modified Prompt: {modified_prompt}")

    image_url = generate_image(modified_prompt, save_path)

    image = load_image_from_url(image_url)
    description = generate_prompt(image)
    # print(f"Generated Description for prompt modifier: {description}")

    return description


def prompt_clarifier(prompt: str, save_path: str):
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
            "content": "Your job is to take user prompts and clarify them so that they are culturally appropriate and reflect the user's intent."
        },
        {
            'role': 'user',
            'content': 'Woman going to the office, comic cartoon style, vivid'
        },
        {
            'content': None,
            'refusal': None,
            'role': 'assistant',
            'function_call': None,
            'tool_calls': [
                {
                    'id': 'call_3I4LS8NyftkAo8Xzf9lMV1bl',
                    'function': {
                        'arguments': '{"question":"What age range is the woman?"}',
                        'name': 'ask_question'
                    },
                    'type': 'function'
                },
                {
                    'id': 'call_GTBcyqWPnw21teDTX8ts4v8z',
                    'function': {
                        'arguments': '{"question":"Where is this taking place?"}',
                        'name': 'ask_question'
                    },
                    'type': 'function'
                },
                {
                    'id': 'call_xNv81gVyVHXWel66GgfdCH1T',
                    'function': {
                        'arguments': '{"question":"Should they being wearing traditional clothing? Or modern Western clothing? or something else?"}',
                        'name': 'ask_question'
                    },
                    'type': 'function'
                }
            ]
        },
        {
            'role': 'tool', 
            'content': '{"answer": "30-35"}',
            'tool_call_id': 'call_3I4LS8NyftkAo8Xzf9lMV1bl'
        },
        {
            'role': 'tool', 
            'content': '{"answer": "New York City"}',
            'tool_call_id': 'call_GTBcyqWPnw21teDTX8ts4v8z'
        },
        {
            'role': 'tool', 
            'content': '{"answer": "Modern clothing"}',
            'tool_call_id': 'call_xNv81gVyVHXWel66GgfdCH1T'
        },
        {
            "role": "assistant",
            "content": "A young, dark haired, 30-35 woman with green eyes walking to work in New York City. She is wearing modern, stylish clothing and has a confident, professional demeanor. The background is a bustling city street with tall buildings and people going about their day."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    while True:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools
        )

        messages.append(json.loads(response.choices[0].message.model_dump_json()))

        question = ""
        if response.choices[0].message.tool_calls:
            for tool_call in response.choices[0].message.tool_calls:
                if tool_call.function.name == "ask_question":
                    question = json.loads(tool_call.function.arguments)["question"]
                    answer = input(f"Answer the question: {question}\n")
                    messages.append({
                        "role": "tool",
                        "content": json.dumps({"answer": answer}),
                        "tool_call_id": tool_call.id
                    })
        else:
            modified_prompt = response.choices[0].message.content
            image_url = generate_image(modified_prompt, save_path)
            image = load_image_from_url(image_url)
            description = generate_prompt(image)
            # print(f"Generated Description for prompt modifier: {description}")

            return description



def ask_chatGPT(prompt, descriptions):
    chat = f"The original prompt is:{prompt}. These are two descriptions of the generated images:\n\n"
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


def compare():
    ethnicities = [
        "African American",	
        "Native American",
        "Hispanic",
        "Asian",
        "Middle Eastern",
        "Indian",
        "Caucasian",
    ]
    for ethnicity in ethnicities:
        prompt = f"{ethnicity} boy going to school, watercolor sketch style, bright colors"

        no_prompt_based_description = no_prompt_modifier(
            prompt=prompt,
            save_path=f"{thisdir}/images/{ethnicity}/no_prompt_modifier.png"
        )

        rule_based_description = rule_based(
            prompt=prompt,
            rules=[
                "Modern urban setting",
                "Traditional clothing styles",
                "Authentic cultural patterns",
                "Rural village landscape",
                "Celebrating cultural festivals",
                "Historical architecture details",
                "Family gathering around a meal",
                "Traditional art techniques",
                "Natural, local environments",
                "Contemporary street fashion",
                "Cultural dance performance",
                "Traditional crafts and artisans",
                "Local market scene",
                "Respectful use of sacred symbols",
                "Culturally inspired interior design",
                "Traditional musical instruments",
                "Religious ceremony in progress",
                "Ethnic jewelry and accessories",
                "Regional food and cuisine",
                "Culturally significant landmarks",
            ],
            save_path=f"{thisdir}/images/{ethnicity}/rule_based.png"
        )

        prompt_based_description = prompt_modifier(
            prompt=prompt,
            save_path=f"{thisdir}/images/{ethnicity}/prompt_modifier.png"
        )

        result = ask_chatGPT(
            prompt=prompt, 
            descriptions=[
                no_prompt_based_description,
                rule_based_description,
                prompt_based_description
            ]
        )
        
        (thisdir / "images" / ethnicity / "result.txt").write_text(result)


def main():
    prompt = "A man going to university, watercolor sketch style, bright colors"
    new_prompt = prompt_clarifier(
        prompt=prompt,
        save_path=f"{thisdir}/images/prompt_clarifier_example.png"
    )
    print(f"New Prompt: {new_prompt}")

if __name__ == "__main__":
    main()

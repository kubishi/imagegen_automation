from dotenv import load_dotenv  # type: ignore
import os
import json
import openai
import requests
import pathlib

load_dotenv()


thisdir = pathlib.Path(__file__).parent.absolute()
client = openai.Client(api_key=os.getenv("OPEN_API_KEY"))


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
    print(f"Prompt: {prompt}")
    print(f"Modified Prompt: {modified_prompt}")
    generate_image(modified_prompt, save_path)


def prompt_modifier(prompt: str, save_path: str):
    messages = [
        {"role": "system", "content": "Your job is to take user prompts and modify them so that they are culturally sensitive and appropriate."},
        {"role": "user", "content": "A native american girl dancing"},
        {"role": "assistant", "content": "A native american girl dancing gracefully, with flowing movement and a joyful expression. She has long, flowing hair that moves with her, and she is wearing a colorful, flowing dress that twirls as she spins. The background is bright and lively, with soft lighting that highlights her movements. Her arms are outstretched, and the scene captures the elegance and energy of dance."},
        {"role": "user", "content": prompt}
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    modified_prompt = response.choices[0].message.content
    print(f"Prompt: {prompt}")
    print(f"Modified Prompt: {modified_prompt}")
    generate_image(modified_prompt, save_path)
    
def main():
    prompt = "Give an image of a native american boy swimming"
    
    rule_based(
        prompt=prompt,
        rules=[
            "no feather's should be involved",
            "no short hair",
            "appropriate modern, business casual attire",
            "appropriate modern, casual beach attire",
            "good background",
            "carrying a school bag"
        ],
        save_path=f"{thisdir}/images/rule_based.png"	
    )

    prompt_modifier(
        prompt=prompt,
        save_path=f"{thisdir}/images/prompt_modifier.png"
    )

if __name__ == "__main__":
    main()

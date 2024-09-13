from dotenv import load_dotenv  # type: ignore
import os
import json
import openai

load_dotenv()


rules = ["no feather's should be involved",
         "no short hair", "appropriate clothing"]
prompt = "Give an image of a native american boy"

client = openai.Client(api_key=os.getenv("OPEN_API_KEY"))


def create_prompt_with_rules(prompt: str, rules: dict) -> str:
    rule_str = json.dumps(rules, indent=2)
    return (
        f"{prompt} : \n{rule_str}"
    )

    funcs = [
        {
            "name": "choosing_rules",
            "description": "Choose rules for prompt which could generate more realistic image",
            "parameters": {
                "type": "object",
                "properties": {
                    "rules": {
                        "type": "list",
                        "description": "Choose from list of rules which could make the prompt better in order to get a more accurate image."
                    }
                },
                "required": ["rules"],
                "additionProperties": False
            }
        }
    ]


modified_prompt = create_prompt_with_rules(prompt, rules)

messages = [
    {"role": "user", "content": modified_prompt}
]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages
)

updated_prompt = response.choices[0].message.content


print(updated_prompt)

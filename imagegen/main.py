from dotenv import load_dotenv  # type: ignore
import os
import json
import openai

load_dotenv()


client = openai.Client(api_key=os.getenv("OPEN_API_KEY"))


def create_prompt_with_rules(prompt: str, rules: dict) -> str:
    return json.dumps({
        "prompt": prompt,
        "rules": rules
    })


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


def main():
    rules = ["no feather's should be involved",
             "no short hair", "appropriate clothing"]

    prompt = "Give an image of a native american boy"

    modified_prompt = create_prompt_with_rules(prompt, rules)

    messages = [
        {"role": "system", "content": "Your job is to take user prompts and decide which rules that are more important should be applied to them so that the image generated is appropriate."},
        {"role": "user", "content": modified_prompt}
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools
    )

    rules = []
    for tool_call in response.choices[0].message.tool_calls:
        if tool_call.function.name == "choose_rules":
            rules.extend(json.loads(tool_call.function.arguments)["rules"])

    modified_prompt = f"{prompt}, " + ", ".join(rules)
    print(f"Modified prompt: {modified_prompt}")


if __name__ == "__main__":
    main()

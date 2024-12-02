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

    prompts_for_day = prompts[(
        day - 1) * prompts_per_day:day * prompts_per_day]
    print(prompts_for_day)

    data: List[Dict[str, Union[str, int]]] = json.loads(
        (thisdir / "data.json").read_text())
    num_examples = min(5, ((day-1)*prompts_per_day*images_per_prompt))
    # get top num_examples examples with most votes
    examples = sorted(
        # Filter out entries with None or missing 'votes'
        [entry for entry in data if entry.get("votes") is not None],
        key=lambda x: x["votes"],
        reverse=True
    )

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


dotenv.load_dotenv()

thisdir = pathlib.Path(__file__).parent.absolute()
client = openai.Client(api_key=os.getenv("OPEN_API_KEY"))

# Replace with your Qualtrics API details
API_TOKEN = os.getenv("QUALTRICS_API_TOKEN")
DATA_CENTER = "iad1"
BASE_URL = f"https://iad1.qualtrics.com/API/v3/survey-definitions"
LIBRARY_ID = "default"

headers = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "X-API-TOKEN": API_TOKEN,
}


def create_survey(survey_name, language, project_category):
    """Create a survey in Qualtrics."""
    # Payload for creating the survey
    payload = {
        "SurveyName": survey_name,
        "Language": language,
        "ProjectCategory": project_category,
    }

    # POST request to create the survey
    response = requests.post(BASE_URL, headers=headers, json=payload)

    # Process the response
    if response.status_code == 200:
        data = response.json()
        survey_id = data.get("result", {}).get("SurveyID", "")
        block_id = data.get("result", {}).get("DefaultBlockID", "")
        print(f"Survey created successfully! Survey ID: {
              survey_id}, Default Block ID: {block_id}")
        return survey_id, block_id
    else:
        print(f"Failed to create survey. Response: {
              response.status_code}, {response.text}")
        return None


def update_survey_questions(survey_id):
    """Add questions to the survey."""
    update_url = f'{BASE_URL}/surveys/{survey_id}/questions'

    # Example question payload
    payload = {
        "questionType": {
            "type": "MC",
            "subType": "SingleAnswer"
        },
        "questionText": "What is your favorite programming language?",
        "choices": {
            "1": {"choiceText": "Python"},
            "2": {"choiceText": "Java"},
            "3": {"choiceText": "C++"},
            "4": {"choiceText": "JavaScript"}
        }
    }

    # API request to add a question
    response = requests.post(update_url, headers=headers, json=payload)

    # Handle the response
    if response.status_code == 200:
        print("Question added successfully!")
    else:
        print("Failed to add question")
        print(f"Response: {response.status_code}, {response.text}")


if __name__ == "__main__":
    # Step 1: Create the survey
    survey_name = "ImageGeneration Survey"
    language = "EN"
    project_category = "CORE"
    survey_id = create_survey(survey_name, language=language,
                              project_category=project_category)

    print(survey_id)
    # Step 2: Add questions (if survey creation was successful)
    # if survey_id:
    #     update_survey_questions(survey_id)

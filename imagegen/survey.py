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
BASE_URL = f"https://{DATA_CENTER}.qualtrics.com/API/v3/surveys"
LIBRARY_ID = "default"

headers = {
    "content-type": "application/json",
    "x-api-token": API_TOKEN,
}


def create_survey() -> str:
    payload = {
        "name": "Image Selection Experiment",
        "isActive": True,
    }
    response = requests.post(BASE_URL, headers=headers, json=payload)
    if response.status_code == 200:
        survey_id = response.json()["result"]["id"]
        print(f"Survey created with ID: {survey_id}")
        return survey_id
    else:
        print(f"Failed to create survey: {
              response.status_code} - {response.json()}")
        return ""


def upload_image(image_path: str) -> str:
    upload_url = f"{BASE_URL}/libraries/{LIBRARY_ID}/files"

    with open(image_path, 'rb') as file:
        files = {'file': file}
        response = requests.post(upload_url, headers=headers, files=files)

    print("Status Code:", response.status_code)
    print("Response Headers:", response.headers)
    print("Response Text:", response.text)

    if response.status_code == 200:
        content_type = response.headers.get("Content-Type", "")
        if "application/json" in content_type:
            try:
                response_data = response.json()
                print(f"Image uploaded successfully: {response_data}")
                return response_data
            except requests.exceptions.JSONDecodeError:
                print("Error: Expected JSON response but received non-JSON content.")
                print("Response Text:", response.text)
                return ""
        else:
            print("Error: Response is not JSON.")
            return ""
    else:
        print(f"Failed to upload image: {response.status_code}")
        return ""


def get_all_images(directory: str) -> List[str]:
    return [
        os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith('.png')
    ]


def add_question_to_survey(survey_id: str, image_ids: List[str]):
    question_payload = {
        "questionText": "Select one of the following images:",
        "questionType": "MC",
        "choices": {str(i+1): {"text": f"<img src='{image_ids[i]}' />"} for i in range(len(image_ids))},
        "choiceOrder": list(range(1, len(image_ids) + 1)),
        "subSelector": "SAVR",
        "selector": "SAVR"
    }
    question_url = f"{BASE_URL}/{survey_id}/questions"
    response = requests.post(
        question_url, headers=headers, json=question_payload)
    if response.status_code == 200:
        question_id = response.json()["result"]["questionId"]
        print(f"Question added to survey {survey_id} with ID: {question_id}")
    else:
        print(f"Failed to add question: {
              response.status_code} - {response.json()}")


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
            "content": f"Modify the following prompt to make it more detailed and culturally appropriate: {prompt}"
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
    }

    prompts = [
        "A native american girl dancing",
        "An indian man walking in Manhattan",
    ]

    image_paths = []
    for prompt in prompts:
        modified_prompt = modify_prompt_basic(prompt)
        savedir = thisdir / f"images/{sanitize_string(prompt)}_modified"
        image_path = savedir / "image.png"
        if not image_path.exists():
            generate_image(modified_prompt, image_path)
        image_paths.append(image_path)

    # Upload images to Qualtrics and gather file IDs
    image_ids = [upload_image(str(image_path))
                 for image_path in image_paths if image_path.exists()]

    # Create survey and add question with image choices
    survey_id = create_survey()
    if survey_id and len(image_ids) >= 3:
        add_question_to_survey(survey_id, image_ids[:3])
    else:
        print("Error: Unable to create survey or insufficient images.")


if __name__ == "__main__":
    main()

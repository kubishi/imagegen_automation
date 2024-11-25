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

API_TOKEN = os.getenv("QUALTRICS_API_TOKEN")
DATA_CENTER = "iad1"
BASE_URL = f"https://{DATA_CENTER}.qualtrics.com/API/v3/surveys"
LIBRARY_ID = "default"

headers = {
    "content-type": "multipart/form-data",
    "x-api-token": API_TOKEN,
}


# Path to the directory containing images
IMAGES_DIR = "/Users/kp/imagegen_automation/imagegen/images/day_1"


def get_all_image_paths(root_dir):

    image_paths = []
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    image_paths.append(os.path.join(folder_path, file))
    return image_paths


def upload_image_to_qualtrics(image_path):

    upload_url = f"{BASE_URL}/libraries/{LIBRARY_ID}/files"
    file_name = os.path.basename(image_path)
    with open(image_path, "rb") as image_file:
        files = {

            "file": (file_name, image_file, "image/png"),
        }
        response = requests.post(upload_url, headers=headers, files=files)
        print(files)
        if response.status_code == 200:

            return response.json()["result"]["id"]
        elif not response.ok:
            print(f"Error: {response.status_code} - {response.text}")
            return None


def create_survey(image_ids):
    """Create a survey with the uploaded images."""
    survey_url = f"{BASE_URL}/surveys"
    payload = {
        "name": "Image Selection Experiment",
        "isActive": True,
        "questions": [],
    }
    for idx, image_id in enumerate(image_ids):
        payload["questions"].append({
            "questionText": f"Question {idx + 1}",
            "questionType": "MC",
            "choices": [{"choiceText": f"Choice {i + 1}"} for i in range(3)],
            "answers": [{"image": {"id": image_id}}],
        })
    response = requests.post(survey_url, headers=headers, json=payload)
    print("RESPONSE" + response.json())
    if response.status_code == 200:
        survey_id = response.json()["result"]["id"]
        print(f"Survey created successfully! Survey ID: {survey_id}")
        return survey_id
    else:
        # print(f"Failed to create survey. Status Code: {response.status_code}")
        # print(f"Response: {response.text}")
        return None


def main():
    # Get all image paths
    image_paths = get_all_image_paths(IMAGES_DIR)
    print(f"Found {len(image_paths)} images.")

    # Upload images to Qualtrics
    image_ids = []
    for image_path in image_paths:
        image_id = upload_image_to_qualtrics(image_path)
        if image_id:
            image_ids.append(image_id)

    # Create survey with the uploaded images
    if image_ids:
        create_survey(image_ids)
    else:
        print("No images uploaded. Survey creation aborted.")


if __name__ == "__main__":
    main()

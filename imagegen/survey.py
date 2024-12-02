import os
import json
import requests

# Configuration
API_TOKEN = os.getenv("QUALTRICS_API_TOKEN")
DATA_CENTER = os.getenv("QUALTRICS_DATA_CENTER")
BASE_URL = f"https://{DATA_CENTER}.qualtrics.com/API/v3"
LIBRARY_ID = os.getenv("QUALTRICS_LIBRARY_ID")
# File to save the uploaded image results
RESULT_FILE = "uploaded_images_result.json"
SURVEY_ID = os.getenv("QUALTRICS_SURVEY_ID")

# Headers for the graphics API request
upload_graphics_headers = {
    "accept": "application/json",
    "X-API-TOKEN": API_TOKEN,
}


def upload_single_image(image_path, image_name, content_type="image/png", folder=None):
    """Upload a single image to the Qualtrics Graphics Library."""
    # Qualtrics API endpoint
    upload_url = f"{BASE_URL}/libraries/{LIBRARY_ID}/graphics"

    # Prepare the multipart/form-data payload
    with open(image_path, "rb") as img_file:
        files = {
            "file": (image_name, img_file, content_type),
        }

        response = requests.post(
            upload_url, headers=upload_graphics_headers, files=files)

    if response.status_code == 200:
        result = response.json()
        print(f"Image '{image_name}' uploaded successfully.")
        return result
    else:
        print(f"Failed to upload image '{
              image_name}'. Status code: {response.status_code}")
        print(f"Response: {response.text}")
        return None


def save_results_to_json(results):
    with open(RESULT_FILE, "w") as file:
        json.dump(results, file, indent=4)
    print(f"Results saved to {RESULT_FILE}")


def create_question_with_images(prompt, image_urls):
    create_question_headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "X-API-TOKEN": API_TOKEN,
    }
    # print(image_urls)
    question_payload = {
        "ChoiceOrder": ["1", "2", "3"],
        "Choices": {
            "1": {
                "Display": f"<img src='{image_urls[0]}' />"  # First image
            },
            "2": {
                "Display": f"<img src='{image_urls[1]}' />"  # Second image
            },
            "3": {
                "Display": f"<img src='{image_urls[2]}' />"  # Third image
            },
        },
        "Configuration": {
            "QuestionDescriptionOption": "UseText",
            "Stack": False,
            "StackItemsInGroups": False
        },
        "QuestionText": prompt,
        "QuestionType": "PGR",  # PGR for progress bar
        "Selector": "DragAndDrop",
        "SubSelector": "NoColumns",
        "Validation": {
            "Settings": {
                "ForceResponse": "OFF",
                "Type": "None"
            }
        },

        "NextChoiceId": 5,
        "NextAnswerId": 4,
        "Groups": [
            "Appropriate",
            "Not Appropriate",
            "Unsure"
        ],
        "DataExportTag": "ImageAccuracyQuestion",
        "NumberOfGroups": 3,

    }

    create_question_url = f"{
        BASE_URL}/survey-definitions/{SURVEY_ID}/questions"
    response = requests.post(
        create_question_url, headers=create_question_headers, json=question_payload)

    if response.status_code == 200:
        question_id = response.json().get("result", {}).get("QuestionID")
        print(f"Question created successfully: {question_id}")
        return question_id
    else:
        print(f"Failed to create question. Status code: {
              response.status_code}")
        print(f"Response: {response.text}")
        return None


def extract_result_ids_for_prompt(response_file, prompt_folder):
    with open(response_file, "r") as file:
        data = json.load(file)

    result_ids = []
    for entry in data:
        if entry["prompt_folder"] == prompt_folder:
            file_id = entry["result"]["result"]["id"]
            result_ids.append(file_id)

    return result_ids


def process_images_and_create_question(image_folder, prompt, response_file, prompt_folder):
    # Extract result IDs for the specified prompt folder
    image_file_ids = extract_result_ids_for_prompt(
        response_file, prompt_folder)

    # Construct image URLs using the extracted result IDs
    image_urls = [f"https://{DATA_CENTER}.qualtrics.com/ControlPanel/Graphic.php?IM={
        file_id}" for file_id in image_file_ids]

    # Create a question with the uploaded images
    if len(image_urls) == 3:
        create_question_with_images(prompt, image_urls)
        print("Question created successfully.")


def main():
    # Base folder containing prompt folders
    base_folder = "/Users/kp/imagegen_automation/imagegen/images/day_1"
    prompt = "Which image is the most accurate representation of the prompt?"
    prompt_folder = "prompt_1"
    # Initialize a list to store results
    # all_results = []
    # image_folder = "/Users/kp/imagegen_automation/imagegen/images/day_1/prompt_0"
    # prompt = "Which image is the most accurate representation of the prompt?"

    # # Process the images and create the question
    # process_images_and_create_question(image_folder, prompt)

    # for prompt_folder in sorted(os.listdir(base_folder)):
    #     prompt_path = os.path.join(base_folder, prompt_folder)
    #     if os.path.isdir(prompt_path) and prompt_folder.startswith("prompt_"):
    #         print(f"Processing folder: {prompt_folder}")

    #         for image_name in sorted(os.listdir(prompt_path)):
    #             if image_name.endswith((".png", ".jpg", ".jpeg")):
    #                 image_path = os.path.join(prompt_path, image_name)
    #                 result = upload_single_image(image_path, image_name)

    #                 if result:
    #                     all_results.append({
    #                         "prompt_folder": prompt_folder,
    #                         "image_name": image_name,
    #                         "result": result,
    #                     })

    # # Save all results to a JSON file
    # save_results_to_json(all_results)

    process_images_and_create_question(
        base_folder, prompt, RESULT_FILE, prompt_folder)


if __name__ == "__main__":
    main()

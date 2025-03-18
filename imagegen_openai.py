import requests
import openai
import pathlib
import dotenv
import os

# Load environment variables
dotenv.load_dotenv()

# Set up OpenAI API key
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

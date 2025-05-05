import base64
from functools import partial
import random
import pathlib
import json
from typing import Any, Callable, List, Dict, Optional, Union
from matplotlib import pyplot as plt
from pydantic import BaseModel
import openai
import os
import dotenv
import shutil
from datetime import datetime
from imagegen_flux import generate_image
import re

# loading environment variables
dotenv.load_dotenv()

# defining the path to the current directory
thisdir = pathlib.Path(__file__).parent.absolute()
client = openai.Client(api_key=os.getenv("OPEN_API_KEY"))

class Rating(BaseModel):
    summary: str
    explanation: str
    rating: int

class RatingAgent(BaseModel):
    user_id: str
    preferences: Dict[str, int]
    system_prompt: str


# def create_and_save_agents(num_agents: int, save_path: pathlib.Path) -> List[RatingAgent]:
#     agents = []

#     for i in range(num_agents):
#         system_prompt, preferences = generate_system_prompt()
#         agent = RatingAgent(
#             user_id=f"agent_{i}",
#             preferences=preferences,
#             system_prompt=system_prompt
#         )
#         agents.append(agent)

    
#     with open(save_path, "w") as f:
#         json.dump([a.dict() for a in agents], f, indent=4, ensure_ascii=False)

#     return agents


def auto_rate(prompt: str,
              image_path: pathlib.Path,
              model: str = "gpt-4o",
              system_prompt: Optional[str] = None) -> Rating:

    image_data = base64.b64encode(image_path.read_bytes()).decode("utf-8")

    messages = []
    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })

    rating_prompt = (
        f"Please evaluate the image generated from this prompt: \"{prompt}\".\n"
        "Provide your assessment of the image quality and cultural appropriateness based on your preferences.\n"
        "Be critical and fair.\n"
        "Format your response as:\n"
        "Explanation: <your explanation>\n"
        "Rating: <a number from 1 to 5>"
    )

    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": rating_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
            ]
        }
    )

    # thisdir.joinpath("test-messages.json").write_text(json.dumps(messages, indent=2, ensure_ascii=False))
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=Rating,
    )

    content: Rating = completion.choices[0].message.parsed

    # explanation_match = re.search(r"Explanation:\s*(.*?)\s*Rating:", content, re.DOTALL | re.IGNORECASE)
    # rating_match = re.search(r"Rating:\s*(\d)", content)

    # explanation = explanation_match.group(1).strip() if explanation_match else content
    # rating = int(rating_match.group(1)) if rating_match else 3

    return content

def modify_prompt(prompt: str, examples: List[Dict[str, str]]) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a prompt modifier that refines text prompts to be detailed but concise, "
                "culturally appropriate, and avoids stereotypes or bias. Use the feedback summaries "
                "to guide your improvements. Also apply negative prompting to steer image generation away "
                "from undesired traits. Keep the modified prompt under 40 words."
            )
        }
    ]

    if not examples:
        examples = [
            {
                "original_prompt": "An asian family having dinner together",
                "modified_prompt": "A modern Asian family sharing dinner at home, avoiding clichés, with casual wear and diverse dishes in a cozy interior. --no traditional stereotypes, --no overexaggerated decor",
                "rating": 3.5,
                "summary": "Pleasant image, but reinforced outdated cultural stereotypes about food, dress, and decor."
            }
        ]

    for example in examples:
        messages.extend([
            {"role": "user", "content": example['original_prompt']},
            {"role": "assistant", "content": example['modified_prompt']},
            {"role": "user", "content": f"Modified Prompt Rating: {example['rating']}. Rating Explanation: {example['summary']}"},
            {"role": "assistant", "content": "Acknowledged"}
        ])

    messages.append({"role": "user", "content": f"Prompt: {prompt}"})

    (thisdir / "messages.json").write_text(json.dumps(messages, indent=4, ensure_ascii=False))

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    return response.choices[0].message.content

def manual_rate(prompt: str, image_path: pathlib.Path) -> Rating:
    print(f"Prompt: {prompt}\nImage Path: {image_path}")
    image = plt.imread(image_path)
    plt.imshow(image)
    plt.axis("off")
    plt.show()

    while True:
        try:
            explanation = input("Please provide an explanation for your rating: ")
            rating = int(input("Please rate the image (1 to 5): "))
            if 1 <= rating <= 5:
                break
            raise ValueError("Rating must be between 1 and 5.")
        except ValueError as e:
            print(e)

    return Rating(explanation=explanation, rating=rating)

def random_rate(prompt: str, image_path: pathlib.Path) -> Rating:
    rating = random.randint(1, 5)
    return Rating(explanation="Randomly Generated", rating=rating)

RatingFunc = Callable[[str, pathlib.Path], Rating]

def summarize_ratings(ratings: List[Rating]) -> str:
    explanations = " ".join([r.explanation for r in ratings])
    summary_prompt = (
        "Give a very short one-sentence summary on the following feedback in a well structured manner "
        "highlighting the key points about the image quality and cultural relevance and why it got this much rating: "
        f"{explanations}"
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": summary_prompt}]
    )
    return response.choices[0].message.content

def rate_images(history: List[Dict[str, str]], num_users: int, rating_funcs: List[Callable[[str, pathlib.Path], Rating]]):
    for i, item in enumerate(history):
        if "ratings" not in item:
            ratings = [rating_funcs[j](item["modified_prompt"], pathlib.Path(item["image_path"])) for j in range(num_users)]
            avg_rating = sum(r.rating for r in ratings) / len(ratings)
            summary = summarize_ratings(ratings)
            history[i].update({
                "ratings": [r.model_dump() for r in ratings],
                "rating": avg_rating,
                "summary": summary
            })

def sanitize_string(s: str) -> str:
    return s.translate(str.maketrans({
        " ": "_", ",": "", ":": "", ";": "", ".": "", "?": "", "!": ""
    }))

def pipeline(
             experiment_dir: pathlib.Path,
             prompts: List[str],
             iterations: int,
             images_per_prompt: int,
             best_n: int,
             num_users: int = 5,
             overwrite: bool = False,
             rating_func: Union[RatingFunc, List[RatingFunc]] = manual_rate):

    history_path = experiment_dir / "history.json"
    images_path = experiment_dir / "images"

    if not isinstance(rating_func, list):
        rating_func = [rating_func] * num_users

    if overwrite:
        if images_path.exists():
            shutil.rmtree(images_path)
        if history_path.exists():
            history_path.write_text("[]")

    history: List[Dict[str, Any]] = []
    if history_path.exists():
        history = json.loads(history_path.read_text())

    best_examples = sorted(history, key=lambda x: x.get("rating", 0), reverse=True)[:best_n]
    for prompt in prompts:
        for iteration in range(iterations):
            print(f"Processing Iteration {iteration + 1} for prompt: {prompt}")
            modified_prompts = [modify_prompt(prompt, best_examples) for _ in range(images_per_prompt)]

            for i, mod_prompt in enumerate(modified_prompts):
                save_path = images_path / f"{sanitize_string(prompt)}/iteration_{iteration}/image_{i}.png"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                if save_path.exists():
                    print(f"Skipping image {save_path}")
                    continue
                generate_image(mod_prompt, save_path)
                history.append({
                    "iteration": iteration,
                    "original_prompt": prompt,
                    "modified_prompt": mod_prompt,
                    "image_path": str(save_path),
                })
                history_path.write_text(json.dumps(history, indent=4, ensure_ascii=False))

            rate_images(history, num_users=num_users, rating_funcs=rating_func)
            history_path.write_text(json.dumps(history, indent=4, ensure_ascii=False))
            best_examples = sorted(history, key=lambda x: x["rating"], reverse=True)[:best_n]

def generate_system_prompt() -> str:
    values_options = {
        "realism": {
            1: "Likes very abstract art and doesn't like photorealism",
            2: "Prefers cartoon images over photorealistic ones",
            3: "Likes realistic and cartoon/abstract art",
            4: "Prefers photorealism",
            5: "Only like hyper-realistic art (photorealism)"
        },
        "traditional": {
            1: "Prefers no traditional cultural elements in the image",
            2: "Slight interest in traditional elements",
            3: "Neutral toward traditional cultural representation",
            4: "Often appreciates traditional cultural elements",
            5: "Strong preference for rich traditional cultural representation"
        },
        "stereotypes": {
            1: "Not sensitive to stereotypes",
            2: "Mildly aware of stereotypes but not bothered",
            3: "Balanced view on stereotypes",
            4: "Prefers avoiding stereotypical representations",
            5: "Highly sensitive to and avoids stereotypes in art"
        },
        "colorful": {
            1: "Prefers muted or monochrome color schemes",
            2: "Leans toward soft, less saturated colors",
            3: "Likes both muted and vivid colors",
            4: "Enjoys bright and vivid colors",
            5: "Strong preference for highly colorful, vibrant art"
        },
        "adherance": {
            1: "Prefers loose or creative interpretation of prompts",
            2: "Values creativity over strict adherence",
            3: "Balanced between prompt following and creativity",
            4: "Prefers closely following the prompt",
            5: "Requires strict and literal adherence to prompts"
        }
    }


    # values = {
    #     "realism": values_options["realism"][1],
    #     "traditional": values_options["traditional"][random.randint(1, 5)],
    #     "stereotypes": values_options["stereotypes"][random.randint(1, 5)],
    #     "colorful": values_options["colorful"][random.randint(1, 5)],
    #     "adherance": values_options["adherance"][random.randint(1, 5)],
    # }

    preferences = {k: random.randint(1, 5) for k in values_options.keys()}
    descriptions = {k: values_options[k][v] for k, v in preferences.items()}

    appendix = (
        "You are a art reviewer with the following preferences: "
        f"{json.dumps(descriptions, ensure_ascii=False)}"
    )
    return appendix, preferences

def create_custom_agents(num_agents: int, shared_trait: str = "realism", shared_value: int = 2) -> List[RatingAgent]:
    values_options = {
        "realism": 2,
        "traditional": [1, 2, 3, 4, 5],
        "stereotypes": [1, 2, 3, 4, 5],
        "colorful": [1, 2, 3, 4, 5],
        "adherance": [1, 2, 3, 4, 5],
    }

    agents = []
    for i in range(num_agents):
        prefs = {}
        for k in values_options:
            if k == shared_trait:
                prefs[k] = shared_value  # everyone agrees on cartoon (realism=2)
            else:
                prefs[k] = random.choice(values_options[k])  # vary others

        descriptions = {
            "realism": {
                1: "Likes very abstract art and doesn't like photorealism",
                2: "Prefers cartoon images over photorealistic ones",
                3: "Likes realistic and cartoon/abstract art",
                4: "Prefers photorealism",
                5: "Only like hyper-realistic art (photorealism)"
            },
            "traditional": {
                1: "Prefers no traditional cultural elements in the image",
                2: "Slight interest in traditional elements",
                3: "Neutral toward traditional cultural representation",
                4: "Often appreciates traditional cultural elements",
                5: "Strong preference for rich traditional cultural representation"
            },
            "stereotypes": {
                1: "Not sensitive to stereotypes",
                2: "Mildly aware of stereotypes but not bothered",
                3: "Balanced view on stereotypes",
                4: "Prefers avoiding stereotypical representations",
                5: "Highly sensitive to and avoids stereotypes in art"
            },
            "colorful": {
                1: "Prefers muted or monochrome color schemes",
                2: "Leans toward soft, less saturated colors",
                3: "Likes both muted and vivid colors",
                4: "Enjoys bright and vivid colors",
                5: "Strong preference for highly colorful, vibrant art"
            },
            "adherance": {
                1: "Prefers loose or creative interpretation of prompts",
                2: "Values creativity over strict adherence",
                3: "Balanced between prompt following and creativity",
                4: "Prefers closely following the prompt",
                5: "Requires strict and literal adherence to prompts"
            }
        }
        description_text = json.dumps({k: descriptions[k][v] for k, v in prefs.items()}, ensure_ascii=False)
        system_prompt = f"You are an art reviewer with the following preferences: {description_text}"
        agents.append(RatingAgent(user_id=f"agent_{i}", preferences=prefs, system_prompt=system_prompt))
    return agents

def create_experiment_dir(root: pathlib.Path) -> pathlib.Path:
    root.mkdir(parents=True, exist_ok=True)

    existing_runs = sorted([
        int(p.name.split("Experiment")[1]) 
        for p in root.glob("Experiment*") 
        if p.is_dir() and p.name.startswith("Experiment") and p.name.split("Experiment")[1].isdigit()
    ])

    next_run_number = max(existing_runs, default=0) + 1
    run_name = f"Experiment{next_run_number:03d}"
    exp_dir = root / run_name
    exp_dir.mkdir(parents=True, exist_ok=False)
    return exp_dir

def run_multiple_experiments(num_runs: int = 10):

    root_dir = thisdir / "experiments"
    prompt_path = thisdir / "prompts.json"
    prompts: List[str] = json.loads(prompt_path.read_text())[:3]  # Use 5 prompts

    for prompt in prompts:
        for run in range(num_runs):
            print(f"\nPrompt: '{prompt}' | Run {run + 1}/{num_runs}")
            experiment_dir = create_experiment_dir(root_dir)
            agents_path = experiment_dir / "agents.json"
            num_agents = 5

            # agents = []
            # auto_rate_funcs = []
            # for i in range(num_agents):
            #     system_prompt, preferences = generate_system_prompt()
            #     agent = RatingAgent(
            #         user_id=f"agent_{i}",
            #         preferences=preferences,
            #         system_prompt=system_prompt
            #     )
            #     agents.append(agent)
            #     auto_rate_funcs.append(
            #         partial(auto_rate, model="gpt-4o-mini", system_prompt=system_prompt)
            #     )

            agents = create_custom_agents(num_agents, shared_trait="realism", shared_value=2)
            auto_rate_funcs = [
                partial(auto_rate, model="gpt-4o-mini", system_prompt=agent.system_prompt)
                for agent in agents
            ]

            with open(agents_path, "w") as f:
                json.dump([a.dict() for a in agents], f, indent=4, ensure_ascii=False)

            pipeline(
                prompts=[prompt],
                iterations=5,
                images_per_prompt=2,
                best_n=3,
                num_users=num_agents,
                overwrite=True,
                rating_func=auto_rate_funcs,
                experiment_dir=experiment_dir
            )

if __name__ == "__main__":
    run_multiple_experiments(num_runs=10)

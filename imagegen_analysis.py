# import json
# import matplotlib.pyplot as plt
# import pathlib
# from collections import defaultdict
# import pandas as pd

# thisdir = pathlib.Path(__file__).parent.resolve()

# """
# This script is used to analyze the data collected from the user study.
# The data is stored in a JSON file, where each item is a dictionary with the following keys
# - iteration: the iteration number
# - original_prompt: the original prompt
# - modified_prompt: the modified prompt
# - image_path: the path to the generated image
# - ratings: a list of dictionaries, each containing the following
#     - user: the user number
#     - rating: the rating given by the user
#     - explanation: the explanation given by the user
# """
# def main():
#     # Load the data
#     history_path = thisdir / "history.json"
#     data = json.loads(history_path.read_text())

#     rows = []
#     for item in data:
#         if "ratings" in item:
#             for i, rating in enumerate(item["ratings"]):

#                 rows.append({
#                     "iteration": item["iteration"],
#                     "original_prompt": item["original_prompt"],
#                     "modified_prompt": item["modified_prompt"],
#                     "image_path": item["image_path"],
#                     "user": i,
#                     "rating": rating["rating"],
#                     "explanation": rating["explanation"]
#                 })

#     df = pd.DataFrame(rows)
#     df_mean = df.groupby(["iteration", "user"])["rating"].mean().reset_index()

#     # Plot the average rating over iterations
#     fig, ax = plt.subplots(figsize=(10, 6))
#     for prompt, group in df_mean.groupby("user"):
#         ax.plot(group["iteration"], group["rating"],
#                 label=f"User {prompt + 1}")

#     plt.ylim(1, 5.5)
#     # only draw integer ticks
#     plt.xticks(range(1, len(df_mean["iteration"].unique())))
#     plt.yticks(range(1, 6))

#     plt.xlabel("Iteration")
#     plt.ylabel("Average Rating")
#     plt.title("Average Rating Over Iterations")
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(thisdir / "ratings.png")


# if __name__ == "__main__":
#     main()

# def rate_images(history: List[Dict[str, str]], num_users: int, rating_func: List[Callable[[str, pathlib.Path], Rating]]):
#     for i, item in enumerate(history):
#         if "ratings" not in item:
#             ratings = [rating_func[j](item["modified_prompt"], pathlib.Path(
#                 item["image_path"])) for j in range(num_users)]
#             avg_rating = sum(r.rating for r in ratings) / len(ratings)
#             summary = summarize_ratings(ratings)
#             history[i].update({

#                 "rating": avg_rating,
#                 "summary": summary
#             })

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from sklearn.metrics.pairwise import euclidean_distances


thisdir = pathlib.Path(__file__).parent.resolve()
experiment_path = thisdir / "experiments" / "Experiment001"
history_path = experiment_path / "history.json"
agents_path = experiment_path / "agents.json"

# Load data
with open(history_path) as f:
    history = json.load(f)
with open(agents_path) as f:
    agents = json.load(f)

# Choose prompt 
selected_prompt = "A Native American boy going to school"
prompt_items = [h for h in history if h['original_prompt'] == selected_prompt]
final_item = max(prompt_items, key=lambda x: x['iteration'])

# Collect ratings from final iteration
ratings = final_item['ratings']
num_users = len(ratings)
user_ratings = [r['rating'] for r in ratings]

# Compute eucledian distance
prefs = pd.DataFrame([
    {
        "user": int(agent["user_id"].split("_")[1]),
        **agent["preferences"]
    }
    for agent in agents
]).set_index("user").sort_index()

avg_pref = prefs.mean().values
user_agreement_raw = np.linalg.norm(prefs.values - avg_pref, axis=1)

# Normalize 
user_agreement = 1 - (user_agreement_raw - user_agreement_raw.min()) / (user_agreement_raw.max() - user_agreement_raw.min() + 1e-8)

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
for i in range(num_users):
    x = user_agreement[i]
    y = user_ratings[i]
    ax.scatter(x, y, s=100, color='black')
    ax.text(x + 0.01, y + 0.05, f"U{i}", fontsize=9)

# Optionally draw a guiding arc/curve
z = np.polyfit(user_agreement, user_ratings, 2)
curve_x = np.linspace(0, 1, 100)
curve_y = np.polyval(z, curve_x)
ax.plot(curve_x, curve_y, linestyle='--', color='gray', alpha=0.6)

ax.set_xlim(0, 1)
ax.set_ylim(min(user_ratings) - 0.5, max(user_ratings) + 0.5)
ax.set_xlabel("User Agreement (Normalized, 1 = high agreement)")
ax.set_ylabel("User's Rating (Final Iteration)")
ax.set_title(f"Per-User Final Rating vs Agreement\n({selected_prompt})")
ax.grid(True)

plt.tight_layout()
plt.savefig(experiment_path / "user_rating_vs_agreement_curve.png")
plt.show()





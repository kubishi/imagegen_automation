import json
import matplotlib.pyplot as plt
import pathlib
from collections import defaultdict
import pandas as pd

thisdir = pathlib.Path(__file__).parent.resolve()


def main():
    # Load the data
    history_path = thisdir / "history.json"
    data = json.loads(history_path.read_text())

    rows = []
    for item in data:
        if "ratings" in item:
            for i, rating in enumerate(item["ratings"]):

                rows.append({
                    "iteration": item["iteration"],
                    "original_prompt": item["original_prompt"],
                    "modified_prompt": item["modified_prompt"],
                    "image_path": item["image_path"],
                    "user": i,
                    "rating": rating["rating"],
                    "explanation": rating["explanation"]
                })

    df = pd.DataFrame(rows)
    df_mean = df.groupby(["iteration", "user"])["rating"].mean().reset_index()

    # Plot the average rating over iterations
    fig, ax = plt.subplots(figsize=(10, 6))
    for prompt, group in df_mean.groupby("user"):
        ax.plot(group["iteration"], group["rating"],
                label=f"User {prompt + 1}")

    plt.ylim(1, 5.5)
    # only draw integer ticks
    plt.xticks(range(1, len(df_mean["iteration"].unique())))
    plt.yticks(range(1, 6))

    plt.xlabel("Iteration")
    plt.ylabel("Average Rating")
    plt.title("Average Rating Over Iterations")
    plt.legend()
    plt.grid(True)
    plt.savefig(thisdir / "ratings.png")


if __name__ == "__main__":
    main()

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from sklearn.metrics.pairwise import euclidean_distances

# Root directory of all experiments
thisdir = pathlib.Path(__file__).parent.resolve()
experiments_root = thisdir / "experiments"
graphs_path = experiments_root / "graphs"
graphs_path.mkdir(exist_ok=True)

results = []

# Iterate through all ExperimentXXX folders
for experiment_path in sorted(experiments_root.glob("Experiment*")):
    history_path = experiment_path / "history.json"
    agents_path = experiment_path / "agents.json"

    if not history_path.exists() or not agents_path.exists():
        print(f"Skipping {experiment_path.name} (missing history or agents)")
        continue

    # Load history and agents
    with open(history_path) as f:
        history = json.load(f)
    with open(agents_path) as f:
        agents = json.load(f)

    # Convert agents to DataFrame
    agent_df = pd.DataFrame([
        {"user_id": agent["user_id"], **agent["preferences"]}
        for agent in agents
    ])
    agent_df["user_index"] = agent_df["user_id"].str.extract(r"(\d+)").astype(int)

    # Group image data by (prompt, run_key)
    runs = {}
    for entry in history:
        prompt = entry["original_prompt"]
        run_key = pathlib.Path(entry["image_path"]).parts[-3]  # folder like "A_Native_American_boy_going_to_school"
        iteration = entry["iteration"]
        runs.setdefault((prompt, run_key), []).append({
            "iteration": iteration,
            "ratings": entry.get("ratings", [])
        })

    # Process each run
    for (prompt, run_id), records in runs.items():
        records_sorted = sorted(records, key=lambda x: x["iteration"])
        final_iter = max(r["iteration"] for r in records)
        final_records = [r for r in records if r["iteration"] == final_iter]

        if not final_records:
            continue

        final_ratings = final_records[0]["ratings"]
        user_ids = list(range(len(final_ratings)))
        user_prefs = agent_df[agent_df["user_index"].isin(user_ids)].sort_values("user_index").drop(columns=["user_id", "user_index"])
        if user_prefs.shape[0] != 5:
            continue

        # Compute user agreement (average distance to centroid)
        centroid = user_prefs.mean().values.reshape(1, -1)
        agreement = euclidean_distances(user_prefs.values, centroid).mean()

        # Compute final image quality (avg rating)
        quality_scores = [r["rating"] for r in final_ratings]
        final_quality = np.mean(quality_scores)

        results.append({
            "prompt": prompt,
            "run_id": experiment_path.name,
            "user_agreement": agreement,
            "final_quality": final_quality
        })
       
# Compile all results into a DataFrame
results_df = pd.DataFrame(results)
print(results_df)

# Plot graphs per prompt
for prompt in results_df["prompt"].unique():
    subset = results_df[results_df["prompt"] == prompt]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(subset["user_agreement"], subset["final_quality"], color='blue', s=100)

    if len(subset) >= 2:
        coeffs = np.polyfit(subset["user_agreement"], subset["final_quality"], 1)
        x_vals = np.array([subset["user_agreement"].min(), subset["user_agreement"].max()])
        y_vals = np.polyval(coeffs, x_vals)
        ax.plot(x_vals, y_vals, linestyle='--', color='red', label="Linear Fit")
        ax.legend()


    ax.set_xlabel("User Agreement (Avg. Euclidean Distance)")
    ax.set_ylabel("Final Image Quality (Avg. Rating at Iteration 10)")
    ax.set_title(f"User Agreement vs Final Image Quality\nPrompt: {prompt}")
    ax.grid(True)
    plt.tight_layout()

    # Save per-prompt graph
    safe_name = prompt.replace(" ", "_").replace("/", "_")
    plt.savefig(graphs_path / f"user_rating_vs_agreement_{safe_name}.png")

# Save results to CSV
results_df.to_csv(experiments_root / "run_level_summary.csv", index=False)

# Print preview
print("\nRun-Level Summary:")
print(results_df.head())




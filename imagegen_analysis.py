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

# Prepare agent preferences
agent_df = pd.DataFrame([
    {
        "user_id": agent["user_id"],
        **agent["preferences"]
    }
    for agent in agents
])
agent_df["user_index"] = agent_df["user_id"].str.extract(r"(\d+)").astype(int)

# Organize by run
runs = {}
for entry in history:
    prompt = entry["original_prompt"]
    run_key = entry["image_path"].split("/")[7]  # Should uniquely identify the run
    iteration = entry["iteration"]
    
    runs.setdefault((prompt, run_key), []).append({
        "iteration": iteration,
        "ratings": entry.get("ratings", [])
    })


results = []

for (prompt, run_id), records in runs.items():
    records_sorted = sorted(records, key=lambda x: x["iteration"])
    final_iter = max(r["iteration"] for r in records)
    final_records = [r for r in records if r["iteration"] == final_iter]
    
    if not final_records:
        continue
    
    # All users in this run
    final_ratings = final_records[0]["ratings"]
    user_ids = list(range(len(final_ratings)))  # assume user_0 to user_4
    user_prefs = agent_df[agent_df["user_index"].isin(user_ids)].sort_values("user_index").drop(columns=["user_id", "user_index"])
    
    if user_prefs.shape[0] != 5:
        continue

    # Compute user similarity
    centroid = user_prefs.mean().values.reshape(1, -1)
    agreement = euclidean_distances(user_prefs.values, centroid).mean()

    # Compute average final quality
    quality_scores = [r["rating"] for r in final_ratings]
    final_quality = np.mean(quality_scores)

    results.append({
        "prompt": prompt,
        "run_id": run_id,
        "user_agreement": agreement,
        "final_quality": final_quality
    })

results_df = pd.DataFrame(results)

# Plot for each prompt
for prompt in results_df["prompt"].unique():
    subset = results_df[results_df["prompt"] == prompt]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(subset["user_agreement"], subset["final_quality"], color='blue', s=100)

    if len(subset) >= 3:
        z = np.polyfit(subset["user_agreement"], subset["final_quality"], 2)
        x_curve = np.linspace(subset["user_agreement"].min(), subset["user_agreement"].max(), 100)
        y_curve = np.polyval(z, x_curve)
        ax.plot(x_curve, y_curve, linestyle='--', color='gray')

    ax.set_xlabel("User Agreement (Avg. Euclidean Distance)")
    ax.set_ylabel("Final Image Quality (Avg. Rating at Iteration 10)")
    ax.set_title(f"User Agreement vs Final Image Quality\nPrompt: {prompt}")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(experiment_path / "user_rating_vs_agreement_curve.png")

print("Run-Level Summary:")
print(results_df)

# Save run level summary to CSV
results_df.to_csv(experiment_path / "run_level_summary.csv", index=False)



import random
import numpy as np
from typing import List
import matplotlib.pyplot as plt
import pandas as pd

def main():
    examples: np.ndarray = np.array([0.0, -1.0, 1.0]) # starts with one 0 example
    points: np.ndarray = np.array([0.0, 0.0, 0.0]) #
    created: np.ndarray = np.array([0, 0, 0]) #
    max_n_examples = 10
    n_images = 3
    n_rounds = 1000
    max_age = 100
    p_new_example = 0.05

    for i in range(n_rounds):
        # remove examples with age > max_age that have less than avg + std points
        # to_remove = np.where((i - created > max_age) & (points < np.mean(points) + np.std(points)))[0]
        # examples = np.delete(examples, to_remove)
        # points = np.delete(points, to_remove)
        # created = np.delete(created, to_remove)

        # get 10 random examples (1 if there are less than 10)
        # get idx of randomly chosen examples
        mode = random.choice([
            # "best", 
            # "worst", 
            "random", 
            # "half-best", 
            # "half-worst"
        ])
        n_examples = len(examples)-1 if len(examples) < max_n_examples else max_n_examples
        if mode == "best":
            idx = np.argsort(points)[-n_examples:]
        elif mode == "worst":
            idx = np.argsort(points)[:n_examples]
        elif mode == "half-best":
            idx_1 = np.argsort(points)[-n_examples//2:]
            idx_2 = random.sample(range(len(examples)), min(n_examples//2, len(examples)))
            idx = np.append(idx_1, idx_2)
        elif mode == "half-worst":
            idx_1 = np.argsort(points)[:n_examples//2]
            idx_2 = random.sample(range(len(examples)), min(n_examples//2, len(examples)))
            idx = np.append(idx_1, idx_2)
        else:
            idx = random.sample(range(len(examples)), min(n_examples, len(examples)))
        
        round_examples = examples[idx]

        if random.random() < p_new_example:
            new_example = np.clip(np.random.normal(loc=np.mean(round_examples), scale=np.std(round_examples)), -1, 1)
            round_examples = np.append(round_examples, new_example)
            examples = np.append(examples, new_example)
            points = np.append(points, 0)
            created = np.append(created, i)

        # Generate n_images image
        for _ in range(n_images):
            image = np.round(np.clip(np.random.normal(loc=np.mean(round_examples), scale=np.std(round_examples)), -1, 1))
            print(image)
            points[idx] += image # * np.clip(1/(i - created[idx]), 0, 1)
    
    # Plot points by example value, color by their index
    df = pd.DataFrame({"Example": examples, "Points": points})
    df["Age"] = -df.index + len(df)
    # keep only examples with age > 500
    # df = df[df["Age"] > 500]
    # remove outliers
    df = df[np.abs(df["Points"] - df["Points"].mean()) <= (3 * df["Points"].std())]
    plt.scatter(df["Example"], df["Points"], c=df["Age"])
    plt.xlabel("Example value")
    plt.ylabel("Points")
    plt.title("Points by example value")
    # legend
    plt.colorbar(label="Age")
    plt.show()


if __name__ == "__main__":
    main()
    
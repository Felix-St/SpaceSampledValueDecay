import pandas as pd
import numpy as np

def get_all_legend_handles_labels(axes):
    """
    Collects all unique legend handles and labels from a list of axes.
    """
    all_handles = []
    all_labels = []

    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for h, lab in zip(handles, labels):
            if lab not in all_labels:  # Avoid duplicates
                all_handles.append(h)
                all_labels.append(lab)

    return all_handles, all_labels


def bin_multiple_seeds(timesteps_list, rewards_list, max_t, bin_size=10000):
    """
    Bin multiple runs and compute mean/std across seeds.

    Args:
        timesteps_list: [seed1_timesteps, seed2_timesteps, ...]
        rewards_list: [seed1_rewards, seed2_rewards, ...]
        bin_size: width of each bin

    Returns:
        DataFrame with columns: bin_center, mean, std
    """
    all_data = []
    for seed_id, (ts, rw) in enumerate(zip(timesteps_list, rewards_list)):
        all_data.append(pd.DataFrame({"seed": seed_id, "timestep": ts, "reward": rw}))

    df = pd.concat(all_data, ignore_index=True)

    bins = np.arange(0, max_t + bin_size, bin_size)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    df["bin_id"] = pd.cut(df["timestep"], bins=bins, labels=False, include_lowest=True)

    # Average within (bin, seed) first
    per_seed = df.groupby(["bin_id", "seed"])["reward"].mean().reset_index()

    # Then mean/std across seeds
    stats = per_seed.groupby("bin_id")["reward"].agg(["mean", "std"]).reset_index()
    stats["bin_center"] = stats["bin_id"].apply(lambda i: bin_centers[int(i)])

    return stats[["bin_center", "mean", "std"]].dropna()


model_to_color = {
    # DQN family: blues (pale -> dark)
    "LimitedDQN": "#9ecae1",
    "DQN":        "#4292c6",
    "DQN_F":      "#08519c",

    # SAC family: purples (pale -> dark)
    "LimitedSAC": "#b9a3e3",
    "SAC":        "#8a63d2",
    "SAC_F":      "#5b2a9e",
    # Others
    "PPO":        "#d9534f",
    "DQN_Reset":  "silver",
    "SAC_Reset":  "silver",
    "DQN_L2Init": "dimgray",
    "SAC_L2Init": "dimgray",
}

model_to_simplified = {
    "DQN_Reset" : "Resetting (Nikishin et al, 2022)",
    "SAC_Reset" : "Resetting (Nikishin et al, 2022)",
    "DQN_L2Init" : "L2ToInit (Kumar et al, 2025)",
    "SAC_L2Init" : "L2ToInit (Kumar et al, 2025)",
    "DQN_F" : "DQN-F (ours)",
    "SAC_F" : "SAC-F (ours)",
}
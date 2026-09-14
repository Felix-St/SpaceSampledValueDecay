"""Combined figure for the hard-exploration section.

    Left half  (50%): learning curves for the files in EXPLORATION_FNAMES,
                      stacked on top of each other, with their own legend.
    Right half (50%): DeepSea scaling (episodes-to-solve vs. N, plus training
                      time on a twin log axis), with its own legend.

The DeepSea part only reads the CSV written by deep_sea_scaling.py, so it does
not need to import that script (and its heavy training dependencies).

    python plot_hard_exploration_combined.py
"""

import csv
import json
import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import offset_copy
from scipy.ndimage import gaussian_filter1d

import gymnasium as gym
import ale_py

from utils.plotting import (
    get_all_legend_handles_labels,
    bin_multiple_seeds,
    model_to_color,
    model_to_simplified,
)

# Imported for their side effect of registering the custom environments.
from environments import nonstationary_lunar_lander, river_swim, choose_box  # noqa: F401
from environments.minigrid_exploration import KeyCorridorSampler  # noqa: F401

gym.register_envs(ale_py)

# ------------------------------------------------------------------ settings
plt.rc("font", **{"family": "serif", "serif": ["times"]})
plt.rc("text", usetex=True)
plt.rc("axes", titlesize=35)
plt.rc("axes", labelsize=25)
plt.rcParams["axes.grid"] = True
plt.rcParams["axes.axisbelow"] = True
matplotlib.rc("xtick", labelsize=30)
matplotlib.rc("ytick", labelsize=30)

OUT_PATH = "./src/imgs/HardExploration_DeepSea.pdf"
FIGSIZE = (22, 9.125)

# Left half
EXPLORATION_FNAMES = ["RiverSwim-v0", "MountainCarContinuous-v0"]
RESULTS_DIR = "results"
SMOOTH_SIGMA = 5
SHARE_X = False         # different envs -> different timestep ranges
LEFT_LEGEND_NCOL = 3    # tune so both legends have the same number of rows

share_x_label_only = True

# Right half
DEEPSEA_CSV = "deep_sea_scaling.csv"
DEEPSEA_ALGOS = ["DQN", "BootDQN", "DQN-F"]
DEEPSEA_SIZES = [5, 10, 15, 20, 30, 40, 50, 60, 70]#, 60, 70, 80]
DEEPSEA_COLORS = {"DQN": "blue", "DQN-F": "orange", "PPO": "red", "BootDQN": "green"}
RIGHT_LEGEND_NCOL = 3    # 5 entries -> 3 + 2 rows, same as the left legend
BUDGET_LW = 8           # dotted budget line (plot + legend)
TIME_MARKERSIZE = 16    # training-time squares (plot + legend)
MEAN_MARKERSIZE = 18    # episodes-to-solve circles (plot + legend)


# ------------------------------------------------------------------ left: learning curves
def get_bin_size(env_str):
    env = gym.make("MiniGrid-KeyCorridorS4R3-v0" if "KeyCorridor" in env_str else env_str)
    try:
        return 2 * env._max_episode_steps
    except Exception:
        pass
    try:
        return 2 * env.env.unwrapped.max_steps
    except Exception:
        return 2 * env.env.unwrapped.horizon
    finally:
        env.close()


def plot_exploration(axs, fnames):
    for ax, fname in zip(axs, fnames):
        with open(os.path.join(RESULTS_DIR, fname + ".json")) as f:
            data = json.load(f)

        env_str = data["environment"]
        bin_size = get_bin_size(env_str)
        total_timesteps = data["total_timesteps"]

        for key in data["models"]:
            if "RND" in key:
                continue
            result = bin_multiple_seeds(
                data[key]["recorded_timesteps"],
                data[key]["reward_history"],
                max_t=total_timesteps,
                bin_size=bin_size,
            )
            mean = gaussian_filter1d(result["mean"], sigma=SMOOTH_SIGMA)
            std = gaussian_filter1d(result["std"], sigma=SMOOTH_SIGMA)
            color = model_to_color[key[:-3]]
            simplified = model_to_simplified.get(key[:-3], key[:-3])

            # Same conventions as plot_seosnsrl.py: "-F" variants on top,
            # plain baselines dashed, "Limited" variants dotted.
            z = 10 if "-F" in simplified else 2
            if simplified in ("DQN", "SAC"):
                ls = "--"
            elif "Limited" in simplified:
                ls = ":"
            else:
                ls = "-"

            ax.fill_between(
                result["bin_center"],
                mean - 0.5 * std,
                mean + 0.5 * std,
                alpha=0.15,
                color=color,
                zorder=z - 1,
            )
            ax.plot(
                result["bin_center"],
                mean,
                color=color,
                label=simplified,
                alpha=0.9,
                linewidth=2,
                zorder=z,
                linestyle=ls,
            )

        ax.text(
            0.04,
            0.045,
            env_str,
            transform=ax.transAxes,
            fontsize=25,
            verticalalignment="bottom",
            horizontalalignment="left",
            bbox=dict(boxstyle="round", facecolor="cyan", edgecolor="black", alpha=0.7),
            zorder=200,  # keep the env label above the zorder-10 curves
        )
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    axs[-1].set_xlabel(r"Timestep $t$")
    if not SHARE_X and not share_x_label_only:
        for ax in axs[:-1]:
            ax.set_xlabel(r"Timestep $t$")

    handles, labels = get_all_legend_handles_labels(axs)
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    return list(handles), list(labels)


# ------------------------------------------------------------------ right: DeepSea
def load_results(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def plot_deepsea(ax, csv_path, algos, sizes):
    rows = load_results(csv_path)
    if not rows:
        sys.exit(f"No results in {csv_path}")
    budget = max(int(r["budget"]) for r in rows)
    present = [a for a in algos if any(r["algo"] == a for r in rows)]

    ax2 = ax.twinx()
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    spread = 0.9
    offsets = np.linspace(-spread, spread, len(present)) if len(present) > 1 else [0.0]

    print(f"\n{'algo':<8}{'N':>4}{'solved':>8}{'mean':>10}{'std':>10}{'min':>10}")
    for algo, dx in zip(present, offsets):
        color = DEEPSEA_COLORS.get(algo, "k")
        speed_x, speed_y = [], []
        mean_x, mean_y = [], []
        for n in sizes:
            runs = [r for r in rows if r["algo"] == algo and int(r["size"]) == n]
            if not runs:
                continue
            vals = np.array(
                [float(r["solved_at"]) if r["solved_at"] else budget for r in runs]
            )
            n_solved = sum(bool(r["solved_at"]) for r in runs)
            all_solved = n_solved == len(runs)
            ax.errorbar(
                n + dx,
                vals.mean(),
                yerr=vals.std(),
                fmt="o",
                color=color,
                markerfacecolor=color if all_solved else "white",
                markeredgecolor="black" if all_solved else color,
                markersize=MEAN_MARKERSIZE,
                capsize=8,
                elinewidth=1.8,
                capthick=1.8,
            )
            mean_x.append(n + dx)
            mean_y.append(vals.mean())
            mins = [float(r["train_time"]) / 60 for r in runs if r.get("train_time")]
            if mins:
                speed_x.append(n + dx)
                speed_y.append(np.mean(mins))
            print(
                f"{algo:<8}{n:>4}{f'{n_solved}/{len(runs)}':>8}"
                f"{vals.mean():>10.0f}{vals.std():>10.0f}"
                f"{np.mean(mins) if mins else float('nan'):>10.1f}"
            )
        # thin solid line through the means, drawn behind the markers
        ax.plot(mean_x, mean_y, color=color, lw=1.5, alpha=0.7, zorder=1)
        ax2.plot(
            speed_x, speed_y,
            ls="--", marker="s", markersize=TIME_MARKERSIZE, lw=2, color=color, alpha=0.4,
        )

    ax.axhline(budget, color="black", ls=":", lw=BUDGET_LW, zorder=0.5)
    ax.set_ylim(bottom=1, top=budget + 1000)
    ax.set_xticks(sizes)
    ax.set_xlabel("DeepSea size $N$")
    ax.set_ylabel("episodes to solve")
    ax.grid(True, which="major", alpha=0.4)
    ax2.set_yscale("log")
    ax2.set_ylabel("training time (min)")
    ax2.grid(False)

    handles = [
        Line2D([], [], color=DEEPSEA_COLORS.get(a, "k"), marker="o", ls="-", lw=1.5,
               markersize=MEAN_MARKERSIZE, label=a)
        for a in present
    ]
    # Explain the marker conventions in the legend as well.
    handles += [
        Line2D([], [], color="gray", marker="s", ls="--", markersize=TIME_MARKERSIZE,
               lw=2, alpha=0.6, label="training time"),
        Line2D([], [], color="black", ls=":", lw=BUDGET_LW,
               label=f"budget ({budget:,} episodes)"),
    ]
    return handles


# ------------------------------------------------------------------ legend helper
def centered_legend(ax, handles, labels, ncol, pad=6, row_gap=4,
                    row_height=None, **kwargs):
    """Legend above `ax` with at most `ncol` entries per row, each row centered.

    Matplotlib always left-aligns an incomplete last row, so instead we draw
    one single-row legend per row and stack them (bottom row first).

    All offsets are in points (not axes fractions), so two legends above axes
    of different heights sit at exactly the same distance from their axes.
    `row_height` (points) forces the row spacing, e.g. to copy it from another
    legend; if None, each row's measured height is used.

    Returns (legends, measured row heights in points).
    """
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    rows = [(handles[i:i + ncol], labels[i:i + ncol])
            for i in range(0, len(handles), ncol)]
    dy = pad
    legends, heights = [], []
    for h, l in reversed(rows):
        trans = offset_copy(ax.transAxes, fig=fig, y=dy, units="points")
        leg = ax.legend(
            h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0),
            bbox_transform=trans, ncol=len(h), borderpad=0.1, **kwargs,
        )
        ax.add_artist(leg)  # keep it when the next ax.legend() is created
        leg.set_clip_on(False)  # add_artist clips to the axes; legend sits outside
        legends.append(leg)
        measured = leg.get_window_extent(renderer).height * 72 / fig.dpi
        heights.append(measured)
        dy += (measured if row_height is None else max(row_height, measured)) + row_gap
    return legends, heights


# ------------------------------------------------------------------ main
def main():
    # One shared 2x2 grid: the left column holds the two stacked panels, the
    # DeepSea panel spans both rows of the right column. Because all axes live
    # in the same grid, constrained layout aligns the top of the upper-left
    # panel with the top of the DeepSea panel, and the bottom of the
    # lower-left panel with its bottom.
    fig = plt.figure(figsize=FIGSIZE, layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1])

    ax_top = fig.add_subplot(gs[0, 0])
    ax_bot = fig.add_subplot(gs[1, 0], sharex=ax_top if SHARE_X else None)
    if SHARE_X:
        ax_top.tick_params(labelbottom=False)
    ax_ds = fig.add_subplot(gs[:, 1])

    # Left: stacked learning curves
    handles, labels = plot_exploration([ax_top, ax_bot], EXPLORATION_FNAMES)
    fig.supylabel("cumulative reward\nper episode (binned)", fontsize=25)
    left_legends, left_heights = centered_legend(
        ax_top, handles, labels, ncol=LEFT_LEGEND_NCOL,
        fontsize=25, frameon=False,
    )
    # Thicker legend lines so the dashed/dotted styles are readable
    # (as in plot_seosnsrl.py).
    for leg in left_legends:
        for line in leg.get_lines():
            line.set_linewidth(4)

    # Right: DeepSea scaling
    ds_handles = plot_deepsea(ax_ds, DEEPSEA_CSV, DEEPSEA_ALGOS, DEEPSEA_SIZES)
    # Reuse the left legend's row height so both legends are equally tall.
    centered_legend(
        ax_ds, ds_handles, [h.get_label() for h in ds_handles],
        ncol=RIGHT_LEGEND_NCOL, row_height=max(left_heights),
        fontsize=25, frameon=False,
    )

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    print(f"\nSaved {OUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
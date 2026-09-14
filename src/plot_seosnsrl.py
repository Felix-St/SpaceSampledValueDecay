import matplotlib
import matplotlib.pyplot as plt

from utils.plotting import (
    get_all_legend_handles_labels,
    bin_multiple_seeds,
    model_to_color,
    model_to_simplified,
)
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

import json

# Needed to read max steps per env, which requires them to be registered
from environments import nonstationary_lunar_lander, river_swim, choose_box

import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

# Plotting Settings
plt.rc("font", **{"family": "serif", "serif": ["times"]})
plt.rc("text", usetex=True)

plt.rc("axes", titlesize=25)
plt.rc("axes", labelsize=18)

plt.rcParams["axes.grid"] = True
plt.rcParams["axes.axisbelow"] = True

matplotlib.rc("xtick", labelsize=24)
matplotlib.rc("ytick", labelsize=24)


cmt = "NSRL" # "Stationary" for the ablation and "NSRL" for the main eval

if cmt == "NSRL":
    # Non Stationary
    fnames = [
        "CartPole-v1",
        "MountainCar-v0",
        "ChooseBoxEnv-v0",
        "InvertedPendulum-v5",
        "MultiPadLunarLanderDiscrete",
        "MultiPadLunarLander",
    ]

    ncol = 3

elif cmt == "Stationary":
    # Stationary Ablation
    fnames = [
        "CartPole-v1_NoDrift",
        "MountainCar-v0_NoDrift",
        "ChooseBoxEnv-v0_NoDrift",
        "InvertedPendulum-v5_NoDrift",
        "MultiPadLunarLanderDiscr_NoDrift",
        "MultiPadLunarLander_NoDrift",
    ]

    ncol = 4


axins = None
axinssac = None


# Constrained layout reserves exactly the space the figure legend needs,
# so no separate legend axes / gridspec row is required.
fig, axs = plt.subplots(3, 2, figsize=(15, 15.125), layout="constrained")
axs = axs.flatten()  # same row-major order as before


complete_env_str = ""

for i, fname in enumerate(fnames):
    res = open("results/" + fname + ".json")
    tmp_data = json.load(res)
    res.close()

    s_names = tmp_data["models"]

    env_str = tmp_data["environment"]

    if any("SAC" in eintrag for eintrag in tmp_data["models"]):
        try:
            sde = tmp_data[list(tmp_data.keys())[0]]["config"]["use_sde"]

            env_str_label = env_str + " (use_sde:" + str(sde) + ")"
        except:
            env_str_label = env_str
    else:
        env_str_label = env_str
    env_str_label = env_str

    complete_env_str = complete_env_str + "_" + env_str

    tmp = gym.make(env_str)

    try:
        bin_size = 5 * tmp._max_episode_steps
    except:
        try:
            bin_size = 5 * tmp.env.unwrapped.max_steps
        except:
            raise Exception("Unknown max steps")
            

    total_timesteps = tmp_data["total_timesteps"]

    for key in s_names:
        recorded_timesteps = tmp_data[key]["recorded_timesteps"]
        recorded_rewards = tmp_data[key]["reward_history"]


        result = bin_multiple_seeds(
            recorded_timesteps,
            recorded_rewards,
            max_t=total_timesteps,
            bin_size=bin_size,
        )

        from scipy.ndimage import gaussian_filter1d

        smoothed_mean = gaussian_filter1d(result["mean"], sigma=10)
        smoothed_std = gaussian_filter1d(result["std"], sigma=10)



        
        simplified = model_to_simplified.get(key[:-3], key[:-3])
        z = 10 if "-F" in simplified else 2
        

        if simplified in ("DQN", "SAC"):
            ls = "--"
        elif "Limited" in simplified:
            ls = ":"
        else:
            ls = "-"

        axs[i].fill_between(
            result["bin_center"],
            smoothed_mean - 0.5 * smoothed_std,
            smoothed_mean + 0.5 * smoothed_std,
            alpha=0.175,
            color=model_to_color[key[:-3]],
            zorder=z - 1,
        )
        axs[i].plot(
            result["bin_center"],
            smoothed_mean,
            color=model_to_color[key[:-3]],
            label=simplified,
            alpha=0.95,
            linewidth=2,
            zorder=z,
            linestyle=ls,
        )

    x_pos = 0.025

    y_pos = 0.045

    
    if "SAC" in str(s_names) and env_str_label == "MultiPadLunarLander":
        env_str_label = env_str_label + " (cont.)"

    if "DQN" in str(s_names) and env_str_label == "MultiPadLunarLander":
            env_str_label = env_str_label + " (discr.)"

    axs[i].text(
        x_pos,
        y_pos,  
        env_str_label,
        transform=axs[i].transAxes,
        fontsize=18,
        verticalalignment="bottom",
        horizontalalignment="left",
        bbox=dict(
        boxstyle="round", facecolor="cyan", edgecolor="black", alpha=0.7),
        zorder=200,
    )

    if i == 4 or i == 5:
        axs[i].set_xlabel(r"Timestep $t$",fontsize=25)

    if i % 2 == 0:
        axs[i].set_ylabel("cumulative reward \n per episode (binned)",fontsize=25)


handles, labels = get_all_legend_handles_labels(axs)


leg = fig.legend(
    handles,
    labels,
    fontsize=25,
    loc="outside upper center",  # above the grid, no gap to fill (mpl >= 3.7)
    ncol=ncol,
    frameon=False,
    borderpad=0.1,  # invisible padding inside the (frameless) legend box
    borderaxespad=0,
)

for line in leg.get_lines():
    line.set_linewidth(4)

# Gap between legend and top row = 2 * h_pad (inches). Lower it to tighten more.
fig.get_layout_engine().set(h_pad=0.02)

fig.align_ylabels(axs[0::2])

fig.draw_without_rendering()
fig.set_layout_engine("none")
to_fig = fig.transFigure.inverted()
legend_top = leg.get_window_extent().transformed(to_fig).y1
plots_center = 0.5 * (axs[0].get_position().x0 + axs[1].get_position().x1)
leg.set_bbox_to_anchor((plots_center, legend_top), transform=fig.transFigure)


env_name_safe = complete_env_str.replace("/", "_")

plt.savefig("./src/imgs/" + "SEOS_" + cmt +  ".pdf", bbox_inches="tight")
plt.show()
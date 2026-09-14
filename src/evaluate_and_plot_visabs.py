import os
import numpy as np, torch, matplotlib.pyplot as plt, gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
from models.custom_agents import DQN_F, DoubleDQN
import matplotlib
from utils.model_configuration import experiment_settings, get_tun_par
from evaluate_nonstgym_parallel import build_env

import torch as th
import matplotlib.patheffects as pe

from stable_baselines3.common.vec_env import VecNormalize, unwrap_vec_normalize

plt.rc("font", **{"family": "serif", "serif": ["times"]})
plt.rc("text", usetex=True)
plt.rc("axes", titlesize=26)
plt.rc("axes", labelsize=24)
plt.rcParams["axes.grid"] = True
plt.rcParams["axes.axisbelow"] = True
matplotlib.rc("xtick", labelsize=24)
matplotlib.rc("ytick", labelsize=24)

benchmark = "VisualAbstractAppendixLong" 

if benchmark == "VisualAbstract":
    configurations = [
        {
            "policy": "MlpPolicy",
            "learning_rate": 4e-3,
            "batch_size": 128,
            "buffer_size": 10000,
            "gamma": 0.98,
            "target_update_interval": 600,
            "train_freq": 16,
            "gradient_steps": 8,
            "exploration_final_eps": 0.07,
            "policy_kwargs":dict(
                optimizer_class=th.optim.AdamW,
                optimizer_kwargs=dict(weight_decay=1e-4),
                net_arch=[256, 256]
            ),
            "exploration_fraction": 0.2, #2,
            "verbose": 0,
        },
        {
            "policy": "MlpPolicy",
            "policy_kwargs":dict(net_arch=[256, 256]),
            "learning_rate": 4e-3,
            "batch_size": 128,
            "buffer_size": 10000,
            "gamma": 0.98,
            "target_update_interval": 600,
            "train_freq": 16,
            "gradient_steps": 8,
            "exploration_final_eps": 0.07,
            "exploration_fraction": 0.2,
            "verbose": 0,
            "forget_ratio": 0.05,
            "eta_forgetting": 0.0015,
            "baseline": 0,
            "world_model": "env",
        }
    ]
    models = [DQN, DQN_F]
    labels = ["DQN", "DQN-F"]
elif benchmark == "VisualAbstractAppendix":
    configurations = [
        {
            "policy": "MlpPolicy",
            "learning_rate": 4e-3,
            "batch_size": 128,
            "buffer_size": 10000,
            "gamma": 0.98,
            "target_update_interval": 600,
            "train_freq": 16,
            "gradient_steps": 8,
            "exploration_final_eps": 0.07,
            "policy_kwargs":dict(
                optimizer_class=th.optim.AdamW,
                optimizer_kwargs=dict(weight_decay=1e-4),
                net_arch=[256, 256]
            ),
            "exploration_fraction": 0.2, #2,
            "verbose": 0,
        },
        {
            "policy": "MlpPolicy",
            "policy_kwargs":dict(net_arch=[256, 256]),
            "learning_rate": 4e-3,
            "batch_size": 128,
            "buffer_size": 10000,
            "gamma": 0.98,
            "target_update_interval": 600,
            "train_freq": 16,
            "gradient_steps": 8,
            "exploration_final_eps": 0.07,
            "exploration_fraction": 0.2,
            "verbose": 0,
            "forget_ratio": 0.05,
            "eta_forgetting": 0.0015,
            "baseline": 0,
            "world_model": "env",
        }
    ]
    models = [DoubleDQN, DQN_F] 
    labels = ["DoubleDQN", "DQN-F"]
elif benchmark == "VisualAbstractAppendixLong":
    configurations = [
        {
            "policy": "MlpPolicy",
            "learning_rate": 4e-3,
            "batch_size": 128,
            "buffer_size": 10000,
            "gamma": 0.98,
            "target_update_interval": 600,
            "train_freq": 16,
            "gradient_steps": 8,
            "exploration_final_eps": 0.07,
            "policy_kwargs":dict(
                optimizer_class=th.optim.AdamW,
                optimizer_kwargs=dict(weight_decay=1e-4),
                net_arch=[256, 256],
            ),
            "exploration_fraction": 0.2, #2,
            "verbose": 0,
        },
        {
            "policy": "MlpPolicy",
            "learning_rate": 4e-3,
            "batch_size": 128,
            "buffer_size": 10000,
            "gamma": 0.98,
            "target_update_interval": 600,
            "train_freq": 16,
            "gradient_steps": 8,
            "exploration_final_eps": 0.07,
            "policy_kwargs":dict(
                optimizer_class=th.optim.AdamW,
                optimizer_kwargs=dict(weight_decay=1e-4),
                net_arch=[256, 256],
                activation_fn=th.nn.Tanh,   
            ),
            "exploration_fraction": 0.2, #2,
            "verbose": 0,
        },
        {
            "policy": "MlpPolicy",
            "policy_kwargs":dict(net_arch=[256, 256]),
            "learning_rate": 4e-3,
            "batch_size": 128,
            "buffer_size": 10000,
            "gamma": 0.98,
            "target_update_interval": 600,
            "train_freq": 16,
            "gradient_steps": 8,
            "exploration_final_eps": 0.07,
            "exploration_fraction": 0.2,
            "verbose": 0,
            "forget_ratio": 0.05,
            "eta_forgetting": 0.0015,
            "baseline": 0,
            "world_model": "env",
        }
    ]
    models = [DoubleDQN, DoubleDQN, DQN_F] 
    labels = ["DoubleDQN", "DoubleDQN (tanh)", "DQN-F"]
elif benchmark == "VisualAbstractWithPPO":
    configurations = [
        {
            "normalize": True,
            "policy": "MlpPolicy",
            "n_steps": 16,
            "gae_lambda": 0.98,
            "gamma": 0.99,
            "n_epochs": 4,
            "ent_coef": 0.0,
        },
        {
            "policy": "MlpPolicy",
            "learning_rate": 4e-3,
            "batch_size": 128,
            "buffer_size": 10000,
            "gamma": 0.98,
            "target_update_interval": 600,
            "train_freq": 16,
            "gradient_steps": 8,
            "exploration_final_eps": 0.07,
            "policy_kwargs":dict(
                optimizer_class=th.optim.AdamW,
                optimizer_kwargs=dict(weight_decay=1e-4),
                net_arch=[256, 256]
            ),
            "exploration_fraction": 0.2, #2,
            "verbose": 0,
        },
        {
            "policy": "MlpPolicy",
            "policy_kwargs":dict(net_arch=[256, 256]),
            "learning_rate": 4e-3,
            "batch_size": 128,
            "buffer_size": 10000,
            "gamma": 0.98,
            "target_update_interval": 600,
            "train_freq": 16,
            "gradient_steps": 8,
            "exploration_final_eps": 0.07,
            "exploration_fraction": 0.2,
            "verbose": 0,
            "forget_ratio": 0.05,
            "eta_forgetting": 0.0015,
            "baseline": 0,
            "world_model": "env",
        }
    ]
    models = [PPO, DoubleDQN, DQN_F] 
    labels = ["PPO", "DoubleDQN", "DQN-F"]

else:
    raise Exception("Unknown benchmark variant")


SNAPSHOT_FRACTIONS = [0.33, 0.66, 1.0]

N_ROLLOUTS = 10
# ---------------------------------------------------------------------------

env_to_benchmark = "MountainCar"
settings = experiment_settings[env_to_benchmark]
total_timesteps = settings["env_settings"]["total_timesteps"]
type_wrapper = settings["env_settings"]["type"]
env_full_name = settings["env_name"]

SEED = 42
np.random.seed(SEED)

ROLLOUT_SEEDS = [SEED + 100 * k for k in range(N_ROLLOUTS)]

fractions = sorted(set(SNAPSHOT_FRACTIONS))
snapshot_steps = [max(1, int(round(f * total_timesteps))) for f in fractions]



def rollout(model, seed=SEED, max_steps=2000, *, env=None):
    """One greedy episode. PPO or DQN, with or without VecNormalize."""
    venv = env if env is not None else model.get_env()
    if not isinstance(venv, VecNormalize):
        venv = VecNormalize(venv, norm_obs=False, norm_reward=False)
    venv.training, venv.norm_reward = False, False

    venv.seed(seed)
    obs = venv.reset()
    traj = []
    for _ in range(max_steps):
        traj.append(np.array(venv.envs[0].unwrapped.state))  
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = venv.step(action)
        if dones[0]:
            traj.append(venv.unnormalize_obs(infos[0]["terminal_observation"])[:2])
            break
    return np.array(traj)


def rollout_bundle(model, seeds=ROLLOUT_SEEDS, max_steps=2000, restore=True):
    """A list of greedy episodes, one per seed, from the same policy."""
    trajs = [rollout(model, s, max_steps) for s in seeds]
    if restore:
        env = model.get_env()
        model._last_obs = env.reset()
        if hasattr(model, "_last_episode_starts"):
            model._last_episode_starts = np.ones((env.num_envs,), dtype=bool)
    return trajs


def q_values(model, obs):
    obs = torch.as_tensor(obs, dtype=torch.float32, device=model.device)
    with torch.no_grad():
        q = model.policy.q_net(obs)
    if q.ndim == 3:  # QRDQN-style: (batch, quantiles, actions)
        q = q.mean(1)
    return q.cpu().numpy()


class SnapshotCallback(BaseCallback):
    """Fire `fn(model, row)` the first time num_timesteps passes each threshold."""

    def __init__(self, checkpoints, fn, verbose=0):
        super().__init__(verbose)
        self.pending = sorted(checkpoints, key=lambda t: t[1])
        self.fn = fn

    def _on_step(self):
        while self.pending and self.num_timesteps >= self.pending[0][1]:
            row, _ = self.pending.pop(0)
            self.fn(self.model, row)
        return True


# --- evaluation grid (built once, from the first model's observation space) ---
n = 250
MARGIN = 0.25  # fraction of each range added on both sides
grid = lo = hi = span = elo = ehi = None


def make_grid(observation_space):
    global grid, lo, hi, span, elo, ehi
    lo, hi = observation_space.low, observation_space.high
    span = hi - lo
    elo, ehi = lo - MARGIN * span, hi + MARGIN * span
    xs, ys = np.linspace(elo[0], ehi[0], n), np.linspace(elo[1], ehi[1], n)
    grid = np.stack(np.meshgrid(xs, ys, indexing="ij"), -1).reshape(-1, 2)


# one dict per row, keyed by agent label
V_rows = [dict() for _ in fractions]
traj_rows = [dict() for _ in fractions]


def state_values(model, obs):
    """V(s) for value-based and actor-critic agents alike."""
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=model.device)
    with torch.no_grad():
        if hasattr(model.policy, "predict_values"):   # PPO / A2C: critic head
            v = model.policy.predict_values(obs_t).reshape(-1)
        else:                                          # DQN-style: greedy Q
            q = model.policy.q_net(obs_t)
            if q.ndim == 3:                            # QRDQN: (B, quantiles, A)
                q = q.mean(1)
            v = q.max(1).values
    return v.cpu().numpy()


def snapshot(model, row, label):
    V_rows[row][label] = state_values(model, grid).reshape(n, n).T
    traj_rows[row][label] = rollout_bundle(model)


trained = {}
for i, model in enumerate(models):
    environment = env_full_name

    tunable_params = get_tun_par(environment, seed=SEED)

    env, num_envs = build_env(
        i,
        SEED,
        tunable_params,
        environment,
        configurations,
        type_wrapper,
        env_to_benchmark,
        total_timesteps,
    )

    conf = dict(configurations[i])

    conf.pop("normalize", False)


    m = model(env=env, seed=SEED, **conf, device="auto")
    m.observation_space.seed(SEED)
    m.action_space.seed(SEED)
    if grid is None:
        make_grid(m.observation_space)

    label = labels[i]
    mid = [(r, s) for r, s in enumerate(snapshot_steps) if s < total_timesteps]
    cb = SnapshotCallback(mid, lambda mdl, r, lb=label: snapshot(mdl, r, lb))
    m.learn(total_timesteps=total_timesteps, progress_bar=True, callback=cb)

    for r, s in enumerate(snapshot_steps):  # any threshold at (or past) the end
        if s >= total_timesteps:
            snapshot(m, r, label)
    trained[label] = m



# --- figure: one row per agent, columns = snapshot fractions ---

tcolors = {}
for label in labels:
    if "-F" in label:
        tcolors[label] = "gold"
    else:
        tcolors[label] = "cyan"

nrows, ncols = len(labels), len(fractions)

h_rat = [0.075] + [0.925/nrows] * nrows

fig, axes = plt.subplots(nrows + 1, ncols, height_ratios=h_rat, figsize=(5.7 * ncols, 4.2 * nrows + 0.75), squeeze=False)

for c, frac in enumerate(fractions):
    V = V_rows[c]
    pct = f"{100 * frac:g}"

    for r, label in enumerate(labels):
        ax = axes[r + 1, c]
        im = ax.imshow(
            V[label],
            origin="lower",
            aspect="auto",
            extent=[elo[0], ehi[0], elo[1], ehi[1]],
            cmap="PiYG",
            alpha=0.9
        )

        # mark the true observation-space bounds
        outline = [pe.Stroke(linewidth=2, foreground="k"), pe.Normal()]

        for x in (lo[0], hi[0]):
            ax.axvline(x, color="w", ls="--", lw=2).set_path_effects(outline)
        for y in (lo[1], hi[1]):
            ax.axhline(y, color="w", ls="--", lw=2).set_path_effects(outline)

        ax.add_patch(
            plt.Rectangle((lo[0], lo[1]), span[0], span[1], fill=False, ec="w", lw=3.5,zorder=1000)
        ).set_path_effects([pe.Stroke(linewidth=5.5, foreground="k"), pe.Normal()])

        for j, T in enumerate(traj_rows[c][label]):
            if len(T) == 0:
                continue
            ax.plot(
                T[:, 0],
                T[:, 1],
                color=tcolors[label],
                lw=1.275,
                alpha=0.75,
                zorder=4,
                label=label if j == 0 else None,
            )
            ax.scatter(
                *T[0],
                color=tcolors[label],
                marker="o",
                s=55,
                zorder=5,
                edgecolor="k",
                linewidth=0.3,
            )
            ax.scatter(
                *T[-1],
                color=tcolors[label],
                marker="*",
                s=150,
                zorder=5,
                edgecolor="k",
                linewidth=0.3,
            )

        if r == 0:
            ax.set_title(rf"${pct}\%$ of steps")
        ax.set_xlim(elo[0], ehi[0])
        ax.set_ylim(elo[1], ehi[1])

    


        if r == nrows - 1:
            ax.set_xlabel("Position")
        else:
            ax.tick_params(bottom=False, labelbottom=False)
        if c == 0:
            ax.set_ylabel("Velocity")
        
        else:
            ax.tick_params(left=False, labelleft=False)

        tx = ax.text(
                lo[0] + 0.02 * span[0] + 0.01,
                hi[1] - 0.14 * span[1],
                "in-distribution",
                color="w",
                fontsize=25,
                zorder=10000,
            )

        tx.set_path_effects([pe.withStroke(linewidth=2, foreground="black")])

        tx = ax.text(
            (lo[0]+hi[0])/2 ,
            hi[1] + 0.12 * span[1],
            label,
            color="k",
            fontsize=16,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="w",
                    edgecolor="k", linewidth=1.0),
        )

        tx = ax.text(
            hi[0] + (MARGIN * span[0])/2,
            0,
            "success",
            color="red",
            fontsize=25,
            rotation=90, 
            rotation_mode='anchor',
            ha="center", va="center",
        )
        tx.set_path_effects([pe.withStroke(linewidth=0.5, foreground="black")])

        

        fig.colorbar(im, ax=ax)
        ax.axvline(x=0.5, color="red", linewidth=2)
        ax.axvspan(0.5, ehi[0], color='red', alpha=0.225, lw=0)

# --- top row: one zig-zag arrow spanning all columns ---
gs = axes[0, 0].get_gridspec()
for ax in axes[0, :]:
    ax.remove()
axa = fig.add_subplot(gs[0, :])
axa.set_axis_off()
axa.set_xlim(0, 1)
axa.set_ylim(-1, 1)

zx = [0.035, 0.10, 0.22, 0.34, 0.46, 0.58, 0.70, 0.82, 0.84]
zy = [0.00, 0.50, -0.50, 0.50, -0.50, 0.50, -0.50, 0.50, 0.00]
axa.plot(zx, zy, color="0.25", lw=2.25, solid_capstyle="round", clip_on=False)
axa.annotate(
    "", xy=(0.9125, 0.0), xytext=(0.839, 0.0),
    arrowprops=dict(arrowstyle="-|>", color="0.25", lw=2.25, mutation_scale=25),
    annotation_clip=False,
)

# end labels: training progress at the tail and the head of the arrow
axa.text(
    zx[0] - 0.018, zy[0] - 0.15, r"$t=0$", color="0.25",
    ha="right", va="center", fontsize=20, clip_on=False,
)
axa.text(
    0.9175, -0.15, r"$t=10^{6}$", color="0.25",
    ha="left", va="center", fontsize=20, clip_on=False,
)

axa.text(
    0.47, 0.0, "power decreases",
    ha="center", va="center", fontsize=20,
    bbox=dict(boxstyle="round,pad=0.35", facecolor="w", edgecolor="none"),
)

plt.tight_layout()


out = "./src/imgs/"
out = out + benchmark + ".pdf"

os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, bbox_inches="tight")
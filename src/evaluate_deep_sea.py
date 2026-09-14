"""DeepSea scaling experiment: episodes-to-solve vs. grid size N.

Runs every (algorithm, N, seed) combination and appends each finished run to a
CSV (re-running the script skips runs already in the CSV). Plotting lives in
plot_hard_exploration_combined.py, which reads that CSV.

    python deep_sea_scaling.py                        # run everything

"Solved" = the greedy policy (no exploration noise) reaches the treasure in
`--consecutive` evaluations in a row, evaluated every `--eval-every` training
episodes. Runs that never solve within `--budget` episodes have an empty
`solved_at` in the CSV.
"""

import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from stable_baselines3.common.utils import LinearSchedule
from models.custom_agents import DQN_F
from models.boot_dqn import BootDQN

SIZES = [
    5,
    10,
    15,
    20,
    30,
    40,
    50, 
    60, 
    70]

SEEDS = [0, 1, 2, 3, 4]
ALGOS = ["DQN", "BootDQN", "DQN-F"]  


DQN_F_CLASS = DQN_F
DQN_F_KWARGS = dict(
    learning_rate=1e-3,
    buffer_size=100_000,
    learning_starts=500,
    batch_size=64,
    train_freq=1,
    target_update_interval=500,
    exploration_fraction=0.3,
    exploration_final_eps=0.05,
    baseline=LinearSchedule(start=100, end=1, end_fraction=0.33),
    forget_ratio=0.25,
    eta_forgetting=LinearSchedule(start=1, end=0.005, end_fraction=0.66),
    world_model="env",
)

DQN_KWARGS = dict(
    learning_rate=1e-3,
    buffer_size=100_000,
    learning_starts=500,
    batch_size=64,
    train_freq=1,
    target_update_interval=500,
    exploration_fraction=0.3,
    exploration_final_eps=0.05,
)

# Bootstrapped DQN + randomized priors, bsuite-style settings: 20 heads, prior
# scale 3, every head sees every transition (mask_prob=1), small replay, batch
# 128, one gradient step per env step, target update every 4 steps.
BOOT_DQN_KWARGS = dict(
    n_heads=20,
    prior_scale=3.0,
    mask_prob=1.0,
    learning_rate=1e-3,
    buffer_size=10_000,
    learning_starts=128,
    batch_size=128,
    train_freq=1,
    target_update_interval=4,
    device="cuda",
)
PPO_KWARGS = dict(n_steps=512, batch_size=64, ent_coef=0.01)
SHARED_KWARGS = dict(verbose=0, policy_kwargs=dict(net_arch=[64, 64]))

# ------------------------------------------------------------------ training
def make_model(algo, env, seed):
    from stable_baselines3 import DQN, PPO

    if algo == "PPO":
        return PPO("MlpPolicy", env, seed=seed, **{**SHARED_KWARGS, **PPO_KWARGS})
    if algo == "DQN":
        return DQN("MlpPolicy", env, seed=seed, **{**SHARED_KWARGS, **DQN_KWARGS})
    if algo == "DQN-F":
        return DQN_F_CLASS(
            policy="MlpPolicy", env=env, seed=seed, **{**SHARED_KWARGS, **DQN_F_KWARGS}
        )
    if algo == "BootDQN":
        return BootDQN(
            policy="MlpPolicy",
            env=env,
            seed=seed,
            **{**SHARED_KWARGS, **BOOT_DQN_KWARGS},
        )
    raise ValueError(f"unknown algo {algo}")


def run_one(algo, size, seed, budget, eval_every, consecutive):
    import torch
    from stable_baselines3.common.callbacks import BaseCallback

    from environments.deep_sea import DeepSeaEnv

    torch.set_num_threads(1)  # we parallelise across processes instead

    class SolveTracker(BaseCallback):
        def __init__(self, eval_env):
            super().__init__()
            self.eval_env = eval_env
            self.episodes = 0
            self.streak = 0
            self.first_treasure = None
            self.solved_at = None
            self._hit = False

        def _greedy_episode_hits_treasure(self):
            obs, _ = self.eval_env.reset()
            done, hit = False, False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = self.eval_env.step(int(action))
                hit |= info["treasure"]
                done = terminated or truncated
            return hit

        def _on_step(self):
            self._hit |= bool(self.locals["infos"][0].get("treasure", False))
            if not self.locals["dones"][0]:
                return True
            self.episodes += 1
            if self._hit and self.first_treasure is None:
                self.first_treasure = self.episodes
            self._hit = False
            if self.episodes % eval_every == 0:
                if self._greedy_episode_hits_treasure():
                    self.streak += 1
                    if self.streak >= consecutive:
                        self.solved_at = self.episodes
                        return False
                else:
                    self.streak = 0
            return self.episodes < budget

    # mapping_seed = seed: each seed also gets a different action layout.
    env = DeepSeaEnv(size=size, mapping_seed=seed)
    eval_env = DeepSeaEnv(size=size, mapping_seed=seed)
    model = make_model(algo, env, seed)
    tracker = SolveTracker(eval_env)
    t0 = time.perf_counter()
    model.learn(total_timesteps=budget * size, callback=tracker, progress_bar=True)
    train_time = time.perf_counter() - t0

    return dict(
        algo=algo,
        size=size,
        seed=seed,
        budget=budget,
        episodes_run=tracker.episodes,
        first_treasure=tracker.first_treasure,
        solved_at=tracker.solved_at,
        train_time=round(train_time, 2),
        steps=model.num_timesteps,
    )


# ------------------------------------------------------------------ CSV I/O
FIELDS = [
    "algo",
    "size",
    "seed",
    "budget",
    "episodes_run",
    "first_treasure",
    "solved_at",
    "train_time",
    "steps",
]


def load_results(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def append_result(path, row):
    old = load_results(path)
    if old and set(old[0]) != set(FIELDS):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows({k: r.get(k) or "" for k in FIELDS} for r in old)
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        w.writerow({k: ("" if row[k] is None else row[k]) for k in FIELDS})


# ------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algos", nargs="+", default=ALGOS)
    p.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument(
        "--budget", type=int, default=10000, help="max training episodes per run"
    )
    p.add_argument(
        "--eval-every", type=int, default=10, help="greedy eval every k episodes"
    )
    p.add_argument(
        "--consecutive",
        type=int,
        default=3,
        help="greedy successes in a row to count as solved",
    )
    p.add_argument("--workers", type=int, default=30)
    p.add_argument("--csv", default="deep_sea_scaling.csv")
    args = p.parse_args()

    done = {
        (r["algo"], int(r["size"]), int(r["seed"])) for r in load_results(args.csv)
    }
    jobs = [
        (a, n, s)
        for a in args.algos
        for n in args.sizes
        for s in args.seeds
        if (a, n, s) not in done
    ]
    print(
        f"{len(jobs)} runs to do ({len(done)} already in {args.csv}), {args.workers} workers"
    )

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                run_one, a, n, s, args.budget, args.eval_every, args.consecutive
            ): (a, n, s)
            for a, n, s in jobs
        }
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                row = fut.result()
            except Exception as e:  # keep the sweep going if one run crashes
                print(f"[{i}/{len(jobs)}] FAILED {futures[fut]}: {e!r}")
                continue
            append_result(args.csv, row)
            print(
                f"[{i}/{len(jobs)}] {row['algo']:<6} N={row['size']:<3} seed={row['seed']} "
                f"solved_at={row['solved_at']}"
            )


if __name__ == "__main__":
    main()
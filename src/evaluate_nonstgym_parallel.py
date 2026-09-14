#!/usr/bin/env python
"""
Parallel version of evaluate_nonstgym.py.

Every (model_index, iteration) pair is independent, so the whole grid is
flattened and dispatched to a process pool. Results are written back into the
same object arrays as the serial version, so the output JSON is unchanged.

Usage:
    python evaluate_nonstgym_parallel.py --env Ant --n-jobs 4
    python evaluate_nonstgym_parallel.py --env CartPole --n-jobs 16 --device cpu
    python evaluate_nonstgym_parallel.py --env Pong --n-jobs 3 --device cuda
"""

import os

# Thread limits must be set before numpy / torch are imported. Without this,
# every worker grabs all cores for BLAS and the pool ends up slower than serial.
for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import json
import copy
import warnings
import argparse
import traceback
from datetime import datetime

import numpy as np
import gymnasium as gym
import torch

torch.backends.nnpack.enabled = False

from joblib import Parallel, delayed

from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecNormalize
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO

from environments.nsrl_tracking_wrapper import (
    LifelongNSWrapperSB3ClassicControl,
    LifelongNSWrapperSB3Mujoco,
)
from environments.nsrl_atari import PongBallSpeedNonStationary, FreewayNonStationary
from environments.minigrid_exploration import KeyCorridorSampler

from minigrid.wrappers import OneHotPartialObsWrapper, ImgObsWrapper

from utils.model_configuration import experiment_settings, get_tun_par

# ---------------------------------------------------------------------------
# Unchanged helpers
# ---------------------------------------------------------------------------




class RewardHistoryCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.history = []  # Stores all episodic rewards
        self.recorded_timesteps = []

    def _on_step(self) -> bool:
        # Collect info from the environment at the end of each episode
        for info in self.locals["infos"]:
            if "episode" in info:  # Check if the episode is done
                self.history.append(info["episode"]["r"])
                self.recorded_timesteps.append(self.num_timesteps)
        return True


def serialize(config):

    tmp_config = copy.deepcopy(config)
    for key in config.keys():
        if callable(config[key]):
            tmp_config[key] = config[key].__str__()

    return tmp_config


# ---------------------------------------------------------------------------
# Environment construction (extracted verbatim from the serial loop body)
# ---------------------------------------------------------------------------


def build_env(
    i,
    seed,
    tunable_params,
    environment,
    configurations,
    type_wrapper,
    env_to_benchmark,
    total_timesteps,
):
    """Returns (env, num_envs)."""
    from environments import nonstationary_lunar_lander, river_swim, choose_box

    num_envs = configurations[i].get("n_envs", 1)

    if type_wrapper == "classic":
        env = make_vec_env(
            environment,
            n_envs=num_envs,
            wrapper_class=lambda e: LifelongNSWrapperSB3ClassicControl(
                e,
                tunable_params,
                seed=seed,
                persistent_params=True,
                change_notification=True,
            ),
        )

        if configurations[i].get("normalize", False):
            env = VecNormalize(env)

    elif type_wrapper == "mujoco":
        if environment != "Ant-v5":
            env = make_vec_env(
                environment,
                n_envs=num_envs,
                wrapper_class=lambda e: LifelongNSWrapperSB3Mujoco(
                    e,
                    tunable_params,
                    seed=seed,
                    persistent_params=True,
                    change_notification=True,
                ),
            )
        else:
            env = make_vec_env(
                environment,
                n_envs=num_envs,
                wrapper_class=lambda e: LifelongNSWrapperSB3Mujoco(
                    e,
                    tunable_params,
                    seed=seed,
                    persistent_params=True,
                    change_notification=True,
                ),
                env_kwargs={"include_cfrc_ext_in_observation": False},
            )

        if configurations[i].get("normalize", False):
            env = VecNormalize(env)

    elif type_wrapper == "atari":
        num_envs = 1

        if environment == "ALE/Pong-v5":
            env = make_vec_env(
                environment,
                n_envs=1,
                wrapper_class=lambda e: PongBallSpeedNonStationary(
                    AtariWrapper(e),
                    start_gain=tunable_params["start_gain"],
                    end_gain=tunable_params["end_gain"],
                    schedule=tunable_params["schedule"],
                    horizon=total_timesteps,
                ),
            )

        elif environment == "ALE/Freeway-v5":
            env = make_vec_env(
                environment,
                n_envs=1,
                wrapper_class=lambda e: FreewayNonStationary(
                    AtariWrapper(e),
                    start_gain=tunable_params["start_gain"],
                    end_gain=tunable_params["end_gain"],
                    schedule=tunable_params["schedule"],
                    horizon=total_timesteps,
                ),
            )

        elif environment == "ALE/MontezumaRevenge-v5":
            env = make_vec_env(
                environment,
                n_envs=1,
                env_kwargs={"frameskip": 1},
                wrapper_class=lambda e: AtariWrapper(e),
            )

        else:
            raise Exception("Unsupported Nonst. Atari env")

        # Temporal information
        env = VecFrameStack(env, n_stack=4)

    elif type_wrapper == "none" or type_wrapper == "river" or type_wrapper == "minigrid_exploration":
        num_envs = 1

        if (
            "MultiPadLLDiscrete" in env_to_benchmark
            or "MultiPadLLContinuous" in env_to_benchmark
        ):
            if tunable_params == {}:
                tunable_params = {
                    "drift_prob": 0.0,
                }
            env = gym.make(
                "MultiPadLunarLander-v0",
                continuous=experiment_settings[env_to_benchmark]["env_settings"]["continuous"],
                render_mode="rgbarray",
                drift_prob=tunable_params["drift_prob"],
                seed=seed,
            )
        elif env_to_benchmark == "RiverSwim":
            env = gym.make("RiverSwim-v0")

        elif "ChooseBox" in env_to_benchmark:
            if tunable_params == {}:
                tunable_params = {
                    "drift_timesteps": [],
                }
            env = gym.make(
                "ChooseBoxEnv-v0",
                drift_timesteps=tunable_params["drift_timesteps"],
                seed=seed,
            )
            env = OneHotPartialObsWrapper(env)
            env = ImgObsWrapper(env)
        elif env_to_benchmark == "KeyCorridor":
            env = gym.make("MiniGrid-KeyCorridorS4R3-v0")
            env = OneHotPartialObsWrapper(env)  
            env = ImgObsWrapper(env) 
            env = KeyCorridorSampler(env=env, carry_key_prob = 0.5)
        else:
            raise Exception("Unknown non-wrapped env")

    else:
        raise Exception("Unknown wrapper type")

    return env, num_envs


def extract_drift(env, model, type_wrapper, num_envs):
    """Returns the drift-timestep record for one finished run."""

    if issubclass(model, PPO) and num_envs > 1:
        return "vectorized_thus_diff_per_vecenv"

    if type_wrapper == "atari" or type_wrapper == "river" or type_wrapper == "minigrid_exploration":
        return "all_curr_no_rng"  # env.envs[0].change_points

    if type_wrapper == "none":
        return env.unwrapped.change_points

    if isinstance(env, DummyVecEnv) or isinstance(env, VecNormalize):
        return env.envs[0].change_points

    return env.change_points


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def run_one(i, iteration, env_to_benchmark, disable_drift, base_seed, device, quiet):
    """
    Runs a single (model, seed) job in its own process.

    Everything is re-derived from `env_to_benchmark` inside the worker so that
    only small picklable values cross the process boundary -- model classes and
    configs (which may hold lambdas / schedules) never get serialized.
    """

    torch.set_num_threads(1)

    settings = experiment_settings[env_to_benchmark]
    total_timesteps = settings["env_settings"]["total_timesteps"]
    type_wrapper = settings["env_settings"]["type"]
    models = settings["models"]
    configurations = settings["configurations"]
    env_full_name = settings["env_name"]

    model = models[i]
    env = None

    try:
        reward_cb = RewardHistoryCallback()

        SEED = base_seed + iteration
        np.random.seed(SEED)


        environment = env_full_name
        tunable_params = get_tun_par(environment, seed=SEED)

        if disable_drift:
            tunable_params = {}

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

        if issubclass(model, PPO):
            conf = copy.deepcopy(configurations[i])
            conf.pop("n_envs", None)
            conf.pop("normalize", None)
        else:
            warnings.warn("n_env is ignored for non classic envs")
            conf = configurations[i]

        current_model = model(env=env, seed=SEED, **conf, device=device)

        current_model.observation_space.seed(SEED)
        current_model.action_space.seed(SEED)

        current_model.learn(
            total_timesteps=total_timesteps,
            callback=reward_cb,
            progress_bar=not quiet,
        )

        drift = extract_drift(env, model, type_wrapper, num_envs)

        return {
            "i": i,
            "iteration": iteration,
            "ok": True,
            "reward_history": reward_cb.history,
            "recorded_timesteps": reward_cb.recorded_timesteps,
            "drift_timesteps": drift,
        }

    except Exception:
        # One bad run shouldn't take down a multi-hour sweep.
        tb = traceback.format_exc()
        print(
            f"\n[FAILED] model={model.__name__}({i}) iteration={iteration}\n{tb}",
            flush=True,
        )
        return {
            "i": i,
            "iteration": iteration,
            "ok": False,
            "reward_history": None,
            "recorded_timesteps": None,
            "drift_timesteps": None,
            "error": tb,
        }

    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def resolve_devices(device_arg):
    """Maps job slots onto devices, round-robin across GPUs when using cuda."""

    if device_arg == "cpu":
        return ["cpu"]

    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
        if device_arg == "cpu":
            return ["cpu"]

    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        print("No CUDA devices visible, falling back to CPU.")
        return ["cpu"]

    return [f"cuda:{k}" for k in range(n_gpus)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="Environment to Benchmark", default="MultiPadLLContinuousNoDrift") # ChooseBox
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=20,
        help="Concurrent worker processes. Size against VRAM when on GPU.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Torch device for the policies. MlpPolicy runs are usually faster on cpu.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show per-run progress bars (they interleave badly with n-jobs > 1).",
    )

    args = parser.parse_args()

    env_to_benchmark = args.env

    base_seed = 2721413865  # Using: od -An -N4 -tu4 /dev/urandom | tr -d ' '

    settings = experiment_settings[env_to_benchmark]
    total_timesteps = settings["env_settings"]["total_timesteps"]
    num_iterations = settings["env_settings"]["num_iterations"]
    models = settings["models"]
    configurations = settings["configurations"]
    env_full_name = settings["env_name"]

    # Drift is defined solely by the experiment config (see model_configuration.py).
    disable_drift = settings["env_settings"].get("drift_disabled", False)

    print(env_to_benchmark)
    print(env_full_name)
    print("Drift disabled:", disable_drift)

    names = [cls.__name__ + "(" + str(i) + ")" for i, cls in enumerate(models)]

    reward_history_per_config = np.empty((len(models), num_iterations), dtype=object)
    drift_timesteps_per_config = np.empty((len(models), num_iterations), dtype=object)
    timesteps_per_config = np.empty((len(models), num_iterations), dtype=object)

    # Flatten both dimensions so the pool stays saturated regardless of whether
    # you have many models and few seeds or the other way round.
    jobs = [(i, it) for i in range(len(models)) for it in range(num_iterations)]

    devices = resolve_devices(args.device)
    quiet = not args.progress

    print(
        f"Dispatching {len(jobs)} runs over {args.n_jobs} workers "
        f"on {devices} ({len(models)} models x {num_iterations} iterations)"
    )

    results_list = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=10)(
        delayed(run_one)(
            i,
            it,
            env_to_benchmark,
            disable_drift,
            base_seed,
            devices[job_idx % len(devices)],
            quiet,
        )
        for job_idx, (i, it) in enumerate(jobs)
    )

    n_failed = 0
    for r in results_list:
        reward_history_per_config[r["i"], r["iteration"]] = r["reward_history"]
        timesteps_per_config[r["i"], r["iteration"]] = r["recorded_timesteps"]
        drift_timesteps_per_config[r["i"], r["iteration"]] = r["drift_timesteps"]
        if not r["ok"]:
            n_failed += 1

    if n_failed:
        print(
            f"\nWARNING: {n_failed}/{len(jobs)} runs failed and are null in the JSON."
        )

    # -----------------------------------------------------------------------
    # Output (identical to the serial script)
    # -----------------------------------------------------------------------

    results = {}

    for i, s_model in enumerate(names):
        tmp_dict = {
            "config": serialize(configurations[i]),
            "reward_history": list(reward_history_per_config[i, :]),
            "drift_timesteps": list(drift_timesteps_per_config[i, :]),
            "recorded_timesteps": list(timesteps_per_config[i, :]),
        }
        results[s_model] = tmp_dict

    results["total_timesteps"] = total_timesteps
    results["num_iterations"] = num_iterations
    results["models"] = names
    results["environment"] = env_full_name

    env_name_safe = env_full_name.replace("/", "_")

    os.makedirs("results", exist_ok=True)

    out_path = (
        "results/"
        + env_name_safe
        + "_MultiPadLongDiscr_"
        + str(datetime.now()).replace(":", "_")
        + "_.json"
    )

    def json_default(obj):
        if isinstance(obj, type):
            # obj is class
            return obj.__name__
        if hasattr(obj, "__class__"):
            # instance of some class
            return obj.__class__.__name__
        return str(obj)  # fallback

    with open(out_path, "w") as filepath:
        json.dump(results, filepath, indent=4, default=json_default)

    print("Wrote", out_path)


if __name__ == "__main__":
    main()
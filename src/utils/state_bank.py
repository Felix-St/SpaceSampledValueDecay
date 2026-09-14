#!/usr/bin/env python3
"""
StateBank for Gymnasium envs, written by `reachable_go_explore.py` (always).

    from gym_state_bank import StateBank
    bank = StateBank("reachability_results/CartPole-v1/bank")
    obs, meta = bank.sample(256, by="room")      # obs: (256, *obs_shape), meta: dict of arrays
    env = gym.make("CartPole-v1")
    obs0 = bank.restore(env, i)                  # env is now in the state behind observation i

Bank directory
    obs.npy     (N, obs_dim) float32  flattened observations (memory-mapped when loading)
    meta.npz    cell_id, region, steps_from_start, steps_from_cell, terminal
    states.npy  (N, k) float64        simulator state per obs (only with --save-sim-states)
    states.pkl  list of snapshots     (replay: (seed, actions); deepcopy: env objects)
    info.json   env id, adapter, shapes, horizon, bbox, ...

CLI:
    python gym_state_bank.py inspect --bank reachability_results/CartPole-v1/bank
"""
import argparse
import copy
import json
import os
import pickle

import numpy as np

DISK_FIELDS = ("cell_id", "region", "steps_from_start", "steps_from_cell", "terminal")


# ----------------------------------------------------------------------------------------------
# Snapshot encoding (one format per adapter of reachable_go_explore.py)
# ----------------------------------------------------------------------------------------------
def decode_snapshot(kind, enc, mujoco_sizes=None):
    """Inverse of encode_snapshot (used when rebuilding a bank from a saved archive)."""
    if kind == "classic":
        return np.asarray(enc, np.float64)
    if kind == "mujoco":
        nq, nv, na = mujoco_sizes
        enc = np.asarray(enc, np.float64)
        return (enc[:nq], enc[nq:nq + nv], enc[nq + nv:nq + nv + na] if na else None)
    return enc  # replay: (seed, actions); deepcopy: the env object


def encode_snapshot(kind, snap):
    if kind == "classic":
        return np.asarray(snap, np.float64).ravel()
    if kind == "mujoco":
        qpos, qvel, act = snap
        parts = [qpos, qvel] + ([act] if act is not None else [])
        return np.concatenate(parts).astype(np.float64)
    if kind == "replay":
        seed, actions = snap
        return (int(seed), np.asarray(actions))
    if kind == "deepcopy":
        return snap
    raise ValueError(kind)


# ----------------------------------------------------------------------------------------------
# Writer
# ----------------------------------------------------------------------------------------------
class BankWriter:
    def __init__(self, bank_dir, kind, n_max, obs_dim, info, save_states=True):
        os.makedirs(bank_dir, exist_ok=True)
        self.dir, self.kind, self.info, self.save_states = bank_dir, kind, info, save_states
        self.tmp_path = os.path.join(bank_dir, "obs.tmp.npy")
        self.obs = np.lib.format.open_memmap(self.tmp_path, mode="w+", dtype=np.float32,
                                             shape=(max(n_max, 1), obs_dim))
        self.meta = {k: [] for k in DISK_FIELDS}
        self.states = []
        self.n = 0

    def add(self, obs_flat, snapshot, **meta):
        self.obs[self.n] = obs_flat
        for k in DISK_FIELDS:
            self.meta[k].append(meta[k])
        if self.save_states:
            self.states.append(encode_snapshot(self.kind, snapshot))
        self.n += 1

    def close(self):
        n = self.n
        final_path = os.path.join(self.dir, "obs.npy")
        final = np.lib.format.open_memmap(final_path, mode="w+", dtype=np.float32,
                                          shape=(n, self.obs.shape[1]))
        for i in range(0, n, 65536):
            final[i:i + 65536] = self.obs[i:min(i + 65536, n)]
        final.flush()
        del final
        del self.obs
        os.remove(self.tmp_path)

        np.savez(os.path.join(self.dir, "meta.npz"),
                 **{k: np.asarray(v, bool if k == "terminal" else np.int64) for k, v in self.meta.items()})
        states_file = None
        if self.save_states:
            if self.kind in ("classic", "mujoco"):
                states_file = "states.npy"
                np.save(os.path.join(self.dir, states_file), np.stack(self.states))
            else:
                states_file = "states.pkl"
                with open(os.path.join(self.dir, states_file), "wb") as f:
                    pickle.dump(self.states, f, protocol=pickle.HIGHEST_PROTOCOL)
        info = dict(self.info, n_obs=n, states_file=states_file)
        if n and "reached_min" in info:
            # rollouts in phase 2 can go slightly beyond the phase-1 data: widen the box
            with_bank = np.load(final_path, mmap_mode="r")
            info["reached_min"] = np.minimum(info["reached_min"], with_bank.min(0)).tolist()
            info["reached_max"] = np.maximum(info["reached_max"], with_bank.max(0)).tolist()
        with open(os.path.join(self.dir, "info.json"), "w") as f:
            json.dump(info, f, indent=2)
        return n


# ----------------------------------------------------------------------------------------------
# Loader / sampler
# ----------------------------------------------------------------------------------------------
class StateBank:
    """Random access to collected observations.

    sample(n, by="room")    uniform over regions ("rooms"), then uniform within   [default]
    sample(n, by="cell")    uniform over source cells
    sample(n, by="uniform") uniform over all stored observations
    """

    def __init__(self, bank_dir, in_memory=False):
        self.dir = bank_dir
        with open(os.path.join(bank_dir, "info.json")) as f:
            self.info = json.load(f)
        self.obs = np.load(os.path.join(bank_dir, "obs.npy"), mmap_mode=None if in_memory else "r")
        with np.load(os.path.join(bank_dir, "meta.npz")) as m:
            self.meta = {k: m[k] for k in m.files}
        self.meta["room"] = self.meta["region"]  # Montezuma-compatible name
        self.obs_shape = tuple(self.info["obs_shape"])
        self.kind = self.info["adapter"]
        self.bbox = (np.asarray(self.info["reached_min"]), np.asarray(self.info["reached_max"]))

        self.states = None
        sf = self.info.get("states_file")
        if sf == "states.npy":
            self.states = np.load(os.path.join(bank_dir, sf), mmap_mode=None if in_memory else "r")
        elif sf == "states.pkl":
            with open(os.path.join(bank_dir, sf), "rb") as f:
                self.states = pickle.load(f)

        self.room_keys, inv = np.unique(self.meta["region"], return_inverse=True)
        inv = inv.reshape(-1)
        self.room_keys = [int(r) for r in self.room_keys]
        self._by_room = [np.flatnonzero(inv == i) for i in range(len(self.room_keys))]
        _, cinv = np.unique(self.meta["cell_id"], return_inverse=True)
        cinv = cinv.reshape(-1)
        self._by_cell = [np.flatnonzero(cinv == i) for i in range(cinv.max() + 1)] if len(cinv) else []
        self._cache = {}

    def __len__(self):
        return len(self.obs)

    def room_counts(self):
        return {k: len(v) for k, v in zip(self.room_keys, self._by_room)}

    region_counts = room_counts

    def _groups(self, by, rooms, include_terminal):
        key = (by, None if rooms is None else tuple(sorted(rooms)), include_terminal)
        if key not in self._cache:
            ok = np.ones(len(self), bool) if include_terminal else ~self.meta["terminal"]
            if rooms is not None:
                ok &= np.isin(self.meta["region"], list(rooms))
            if by == "uniform":
                groups = [np.flatnonzero(ok)]
            elif by in ("room", "region"):
                groups = [g[ok[g]] for g in self._by_room]
            elif by == "cell":
                groups = [g[ok[g]] for g in self._by_cell]
            else:
                raise ValueError(f"unknown sampling mode {by!r}")
            groups = [g for g in groups if len(g)]
            if not groups:
                raise ValueError("no observations match the filter")
            self._cache[key] = groups
        return self._cache[key]

    def sample_indices(self, n, by="room", rng=None, rooms=None, include_terminal=False):
        rng = rng if rng is not None else np.random.default_rng()
        groups = self._groups(by, rooms, include_terminal)
        if len(groups) == 1:
            g = groups[0]
            return g[rng.integers(len(g), size=n)]
        picks = rng.integers(len(groups), size=n)
        return np.array([groups[g][rng.integers(len(groups[g]))] for g in picks])

    def sample(self, n, by="room", rng=None, rooms=None, include_terminal=False):
        """Returns obs (n, *obs_shape) float32 and a dict of metadata arrays.
        `rooms` optionally restricts sampling to these region ids."""
        idx = self.sample_indices(n, by, rng, rooms, include_terminal)
        order = np.argsort(idx)  # sorted reads are much faster on a memmap
        out = np.empty((n, self.obs.shape[1]), np.float32)
        out[order] = self.obs[idx[order]]
        meta = {k: v[idx] for k, v in self.meta.items()}
        meta["index"] = idx
        return out.reshape(n, *self.obs_shape), meta

    def in_bbox(self, x):
        """True where (flattened) observations lie inside the reached [min, max] box."""
        x = np.asarray(x, np.float64).reshape(-1, self.obs.shape[1])
        lo, hi = self.bbox
        return np.all((x >= lo) & (x <= hi), axis=1)

    # ------------------------------------------------------------------------------------------
    def restore(self, env, i, keep_time=False):
        """Put `env` (created with gym.make(env_id)) into the state behind observation i and
        return that observation in the env's (unwrapped) observation format.

        keep_time=False: the TimeLimit counter starts at 0 (full episode budget from here).
        keep_time=True : the counter is set to the steps it took to reach this state."""
        import gymnasium as gym

        if self.states is None:
            raise RuntimeError("bank has no simulator states; regenerate with --save-sim-states")
        snap = self.states[i]
        kind = self.kind

        if kind == "replay":
            seed, actions = snap
            env.reset(seed=seed)
            u = env.unwrapped
            for a in actions:
                u.step(a)
        else:
            env.reset()  # initialises all wrappers
            u = env.unwrapped
            if kind == "classic":
                u.state = np.array(snap, np.float64)
                if hasattr(u, "steps_beyond_terminated"):
                    u.steps_beyond_terminated = None
            elif kind == "mujoco":
                nq, nv, na = self.info["mujoco_sizes"]
                snap = np.asarray(snap)
                if na:
                    u.data.act[:] = snap[nq + nv:nq + nv + na]
                u.set_state(snap[:nq], snap[nq:nq + nv])
            elif kind == "deepcopy":
                _replace_unwrapped(env, copy.deepcopy(snap))
            else:
                raise ValueError(kind)

        w = env
        while isinstance(w, gym.Wrapper):
            if isinstance(w, gym.wrappers.TimeLimit):
                w._elapsed_steps = int(self.meta["steps_from_start"][i]) if keep_time else 0
            w = w.env
        return gym.spaces.unflatten(env.unwrapped.observation_space, np.asarray(self.obs[i]))


def _replace_unwrapped(env, new_unwrapped):
    import gymnasium as gym
    w = env
    while isinstance(w, gym.Wrapper):
        if not isinstance(w.env, gym.Wrapper):
            w.env = new_unwrapped
            return
        w = w.env
    raise ValueError("deepcopy restore needs a wrapped env (create it with gym.make)")


# ----------------------------------------------------------------------------------------------
# inspect
# ----------------------------------------------------------------------------------------------
def inspect(args):
    bank = StateBank(args.bank)
    counts = bank.room_counts()
    term = bank.meta["terminal"]
    print(f"{bank.info['env_id']}: {len(bank)} observations, {len(bank._by_cell)} source cells, "
          f"{len(counts)} regions, {term.mean():.1%} terminal, adapter={bank.kind}, "
          f"states={'yes' if bank.states is not None else 'no'}")
    sizes = np.array(sorted(counts.values()))
    print(f"  observations per region: min {sizes.min()}  median {int(np.median(sizes))}  max {sizes.max()}")
    lo, hi = bank.bbox
    for d in range(min(len(lo), 12)):
        print(f"  obs[{d}] reached range [{lo[d]: .4g}, {hi[d]: .4g}]")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if bank.obs.shape[1] >= 2:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.permutation(len(bank))[:20000])
        x = np.asarray(bank.obs[idx])
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.scatter(x[:, 0], x[:, 1], c=bank.meta["region"][idx] % 20, cmap="tab20", s=2, alpha=0.5)
        ax.set_xlabel("obs[0]")
        ax.set_ylabel("obs[1]")
        ax.set_title(f"{bank.info['env_id']} bank (colour = region)")
        fig.tight_layout()
        path = os.path.join(args.bank, "preview.png")
        fig.savefig(path, dpi=120)
        print(f"preview -> {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("inspect", help="print bank statistics and write preview.png")
    i.add_argument("--bank", required=True)
    args = p.parse_args()
    {"inspect": inspect}[args.cmd](args)


if __name__ == "__main__":
    main()
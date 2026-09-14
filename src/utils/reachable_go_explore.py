"""
Go-Explore-style estimation of the *reachable* observation set of Gymnasium
environments, and comparison against `observation_space.sample()`.

For each env:
  1. Warm-up random rollouts -> scale for the cell discretisation.
  2. Go-Explore loop: select cell -> restore simulator state -> explore with
     (sticky) random actions -> add new cells to the archive.
     Every archived cell has a witness trajectory from a real reset state,
     so everything in the archive is provably reachable.
  3. Draw N samples from observation_space.sample() and check whether each lies
     inside the bounding box of reached observations, i.e. within the reached
     [min, max] range in every dimension. Reported both overall and per dim.
     (Cells are only used to drive exploration, not for the coverage metric.)
  4. Save data (npz), metrics (json) and plots (saturation curve, 2-D scatter
     with the reached bounding box drawn in).
  5. Always: Montezuma-style phase 2. Restore every archived cell, store its
     observation plus observations from short random rollouts into
     <out>/<env>/bank/. Load it with gym_state_bank.StateBank and sample().

Usage:
  python reachable_go_explore.py                   # all envs listed in ENVS below
  python reachable_go_explore.py --envs CartPole-v1 MountainCar-v0

Importing:
  from gym_state_bank import StateBank
  obs, meta = StateBank("reachability_results/CartPole-v1/bank").sample(256)
"""
import argparse
import copy
import importlib
import json
import os
import pickle
import time
from types import SimpleNamespace
from utils.state_bank import BankWriter, decode_snapshot, encode_snapshot
import numpy as np
import gymnasium as gym

# --------------------------------------------------------------------------
# Per-env configuration. Anything not listed falls back to DEFAULTS.
#   cell_dims : indices of the *flattened* observation used for cells
#               (None = all). For high-dim envs pick the meaningful ones.
#   bins      : cells per dimension over the scaled range
#   plot_pairs: (i, j) observation-dim pairs for the 2-D scatter plots
# --------------------------------------------------------------------------
DEFAULTS = dict(
    adapter="auto",
    kwargs={},             # extra kwargs for gym.make
    cell_dims=None,
    bins=15,
    iters=10000,
    explore_steps=50,
    repeat_prob=0.75,      # prob. of repeating the previous action (sticky exploration)
    n_start_seeds=20,      # number of distinct reset() states seeded into the archive
    warmup_episodes=20,
    plot_pairs=[(0, 1)],
)

ENV_CONFIGS = {
    "CartPole-v1": dict(adapter="classic", bins=20, plot_pairs=[(0, 1), (2, 3)]),
    "MountainCar-v0": dict(adapter="classic", bins=50, plot_pairs=[(0, 1)]),
    "InvertedPendulum-v5": dict(adapter="mujoco", bins=20, plot_pairs=[(0, 1), (2, 3)]),
    # Ant obs (v5 default, 105 dims): [0] torso z, [1:5] torso quat, [5:13] joint
    # angles, [13:27] velocities, [27:] contact forces. Cells over pose only.
    # cfrc_ext (78 always-zero contact-force dims) is dropped -> 27-dim observation
    "Ant-v5": dict(adapter="mujoco", cell_dims=list(range(13)), bins=6, explore_steps=100,
                   kwargs=dict(include_cfrc_ext_in_observation=False),
                   plot_pairs=[(0, 1), (5, 6)]),
    # LunarLander obs: x, y, vx, vy, angle, angvel, leg1, leg2
    #"LunarLander-v3": dict(adapter="replay", cell_dims=[0, 1, 2, 3, 4, 5], bins=15,
    #                       plot_pairs=[(0, 1), (2, 3)]),
}

# --------------------------------------------------------------------------
# SETTINGS (edit here; not exposed on the command line)
# --------------------------------------------------------------------------
# Envs run when --envs is not given. Unregistered envs are skipped with a message.
ENVS = ["CartPole-v1", "InvertedPendulum-v5", "ChooseBoxEnv-v0", "MountainCar-v0", "Ant-v5",
        "MultiPadLunarLander-v0"]
# Modules imported before running (put the module(s) registering your custom envs here,
# e.g. ["my_package.envs"]); anything passed with --import is added.
IMPORTS = []

SEED = 0
SAVE_ARCHIVE = True       # write archive.pkl so --bank-only can rebuild the bank later
N_UNIFORM = 10_000        # observation_space.sample() draws for the coverage check
MAX_REACHED = 200_000     # reached observations kept for the coverage check
BANK = dict(
    per_cell=10,          # observations stored per archived cell
    rollout_len=50,       # max steps of a random rollout from a cell
    per_rollout=2,        # observations recorded per rollout
    max_attempts=20,      # max rollouts per cell
    max_cells=20_000,     # use a random subset of at most this many cells (0 = all)
    region_factor=4,      # region ("room") = cell index // region_factor
    save_sim_states=False,  # True also stores simulator states (for StateBank.restore)
)


def env_config(env_id):
    """DEFAULTS updated with the ENV_CONFIGS entry whose name matches env_id
    (exact id first, then by name, e.g. 'MultiPadLunarLander-v0' -> 'LunarLander')."""
    cfg = dict(DEFAULTS)
    if env_id in ENV_CONFIGS:
        cfg.update(ENV_CONFIGS[env_id])
        return cfg, env_id
    name = env_id.split("/")[-1].split("-v")[0]
    for key, c in ENV_CONFIGS.items():
        if key.split("-v")[0] in name:
            cfg.update(c)
            return cfg, key
    return cfg, "defaults"


# ============================== adapters ==================================
class BaseAdapter:
    """Steps env.unwrapped directly so the TimeLimit wrapper does not interfere;
    the horizon is enforced by the explorer instead."""
    name = "base"

    def __init__(self, env):
        self.env = env
        self.u = env.unwrapped

    def reset(self, seed):
        obs, _ = self.env.reset(seed=seed)
        self.u = self.env.unwrapped
        return obs

    def step(self, action):
        obs, _r, terminated, _trunc, _info = self.u.step(action)
        return obs, bool(terminated)

    def snapshot(self):
        raise NotImplementedError

    def restore(self, snap):
        raise NotImplementedError


class ClassicStateAdapter(BaseAdapter):
    name = "classic"

    def snapshot(self):
        return np.array(self.u.state, dtype=np.float64).copy()

    def restore(self, snap):
        self.u.state = snap.copy()
        if hasattr(self.u, "steps_beyond_terminated"):  # CartPole
            self.u.steps_beyond_terminated = None


class MujocoAdapter(BaseAdapter):
    name = "mujoco"

    def snapshot(self):
        d = self.u.data
        act = d.act.copy() if d.act.size else None
        return (d.qpos.copy(), d.qvel.copy(), act)

    def restore(self, snap):
        qpos, qvel, act = snap
        if act is not None:
            self.u.data.act[:] = act
        self.u.set_state(qpos, qvel)


class DeepcopyAdapter(BaseAdapter):
    name = "deepcopy"

    def snapshot(self):
        return copy.deepcopy(self.u)

    def restore(self, snap):
        self.u = copy.deepcopy(snap)


class ReplayAdapter(BaseAdapter):
    """State = (seed, action sequence). Works for anything deterministic given
    the reset seed (e.g. Box2D, where the physics world can't be copied)."""
    name = "replay"

    def reset(self, seed):
        self._seed, self._actions = seed, []
        return super().reset(seed)

    def step(self, action):
        self._actions.append(np.copy(action) if isinstance(action, np.ndarray) else action)
        return super().step(action)

    def snapshot(self):
        return (self._seed, tuple(self._actions))

    def restore(self, snap):
        seed, actions = snap
        self.env.reset(seed=seed)
        self.u = self.env.unwrapped
        for a in actions:
            self.u.step(a)
        self._seed, self._actions = seed, list(actions)


ADAPTERS = {c.name: c for c in (ClassicStateAdapter, MujocoAdapter, DeepcopyAdapter, ReplayAdapter)}


def restore_is_deterministic(adapter, flat, action_space, n=40, seed=0):
    """Snapshot after reset + a few steps, run a fixed action sequence twice
    (restoring in between) and check the observations match."""
    try:
        adapter.reset(seed)
        action_space.seed(seed)
        for _ in range(5):
            adapter.step(action_space.sample())
        snap = adapter.snapshot()
        acts = [action_space.sample() for _ in range(n)]

        def run():
            out = []
            for a in acts:
                o, term = adapter.step(a)
                out.append(flat(o))
                if term:
                    break
            return out

        o1 = run()
        adapter.restore(snap)
        o2 = run()
        return len(o1) == len(o2) and all(np.allclose(a, b, atol=1e-5) for a, b in zip(o1, o2))
    except Exception as e:  # noqa: BLE001
        print(f"    [{adapter.name}] restore test raised: {type(e).__name__}: {e}")
        return False


def make_adapter(env, kind, flat, action_space):
    u = env.unwrapped
    if kind == "auto":
        if hasattr(u, "set_state") and hasattr(getattr(u, "data", None), "qpos"):
            candidates = ["mujoco"]
        elif type(u).__module__.startswith("gymnasium.envs.classic_control") and hasattr(u, "state"):
            candidates = ["classic"]
        elif hasattr(u, "world"):  # Box2D
            candidates = ["replay"]
        else:
            candidates = ["deepcopy", "replay"]
    else:
        candidates = [kind]
    for c in candidates:
        adapter = ADAPTERS[c](env)
        if restore_is_deterministic(adapter, flat, action_space):
            return adapter
        print(f"    adapter '{c}' failed the restore-determinism check")
    print(f"    WARNING: no adapter passed the check, using '{candidates[-1]}' anyway")
    return ADAPTERS[candidates[-1]](env)


# ============================ cells & archive ==============================
HUGE = 1e10  # bounds beyond this (e.g. float32 max in CartPole) are treated as infinite


class CellMapper:
    def __init__(self, flat_space, warm_obs, dims, bins):
        self.dims = np.arange(flat_space.shape[0]) if dims is None else np.asarray(dims)
        lo_s = flat_space.low[self.dims].astype(np.float64)
        hi_s = flat_space.high[self.dims].astype(np.float64)
        finite = np.isfinite(lo_s) & np.isfinite(hi_s) & (np.abs(lo_s) < HUGE) & (np.abs(hi_s) < HUGE)
        w = warm_obs[:, self.dims]
        lo_w, hi_w = np.percentile(w, 1, axis=0), np.percentile(w, 99, axis=0)
        span = hi_w - lo_w
        lo_w, hi_w = lo_w - 0.5 * span, hi_w + 0.5 * span  # margin: exploration goes further
        lo, hi = np.where(finite, lo_s, lo_w), np.where(finite, hi_s, hi_w)
        self.lo = lo
        self.width = (hi - lo) / bins
        self.width[self.width <= 1e-8] = 1.0
        # indices are NOT clipped: states beyond the scaled range get their own cells

    def __call__(self, f):
        z = (np.asarray(f, dtype=np.float64)[self.dims] - self.lo) / self.width
        z = np.clip(z, -1e12, 1e12)  # keep absurd uniform samples castable
        return tuple(np.floor(z).astype(np.int64).tolist())


class Archive:
    def __init__(self, cap=1024):
        self.index, self.snaps, self.rep_obs = {}, [], []
        self.n = 0
        self.traj_len = np.zeros(cap, np.int64)
        self.chosen = np.zeros(cap, np.int64)
        self.seen = np.zeros(cap, np.int64)
        self.terminal = np.zeros(cap, bool)

    def _grow(self):
        for name in ("traj_len", "chosen", "seen", "terminal"):
            a = getattr(self, name)
            b = np.zeros(2 * len(a), a.dtype)
            b[: len(a)] = a
            setattr(self, name, b)

    def update(self, key, f, t, terminal, snapshot_fn):
        i = self.index.get(key)
        if i is None:
            if self.n == len(self.traj_len):
                self._grow()
            i = self.n
            self.index[key] = i
            self.snaps.append(snapshot_fn())
            self.rep_obs.append(f)
            self.traj_len[i], self.terminal[i], self.seen[i] = t, terminal, 1
            self.n += 1
            return True
        self.seen[i] += 1
        # prefer non-terminal entries, then shorter trajectories (Go-Explore heuristic)
        better = (self.terminal[i] and not terminal) or (terminal == self.terminal[i] and t < self.traj_len[i])
        if better:
            self.snaps[i], self.rep_obs[i] = snapshot_fn(), f
            self.traj_len[i], self.terminal[i] = t, terminal
        return False

    def select(self, rng, horizon):
        n = self.n
        w = 1.0 / np.sqrt(1 + self.chosen[:n]) + 0.5 / np.sqrt(1 + self.seen[:n])
        w[self.terminal[:n] | (self.traj_len[:n] >= horizon)] = 0.0
        s = w.sum()
        return None if s == 0 else int(rng.choice(n, p=w / s))


class Reservoir:
    def __init__(self, cap, dim, rng):
        self.buf = np.empty((cap, dim), np.float64)
        self.cap, self.n_seen, self.rng = cap, 0, rng

    def add(self, x):
        if self.n_seen < self.cap:
            self.buf[self.n_seen] = x
        else:
            j = self.rng.integers(0, self.n_seen + 1)
            if j < self.cap:
                self.buf[j] = x
        self.n_seen += 1

    def data(self):
        return self.buf[: min(self.n_seen, self.cap)]


# ============================== exploration ================================
def warmup(adapter, flat, action_space, episodes, horizon, seed):
    obs_list = []
    for ep in range(episodes):
        o = adapter.reset(seed + 10_000 + ep)
        obs_list.append(flat(o))
        for _ in range(horizon):
            o, term = adapter.step(action_space.sample())
            obs_list.append(flat(o))
            if term:
                break
    return np.array(obs_list)


def go_explore(adapter, flat, cellmap, action_space, cfg, horizon, rng, max_reached, log_every=500):
    archive = Archive()
    dim = len(flat(adapter.reset(cfg["seed"])))
    res = Reservoir(max_reached, dim, rng)

    for s in range(cfg["n_start_seeds"]):
        f = flat(adapter.reset(cfg["seed"] + s))
        res.add(f)
        archive.update(cellmap(f), f, 0, False, adapter.snapshot)

    history, total_steps, t0 = [], 0, time.time()
    for it in range(cfg["iters"]):
        i = archive.select(rng, horizon)
        if i is None:
            print("    archive exhausted (all cells terminal / at horizon)")
            break
        archive.chosen[i] += 1
        adapter.restore(archive.snaps[i])
        t = int(archive.traj_len[i])
        a = action_space.sample()
        for _ in range(cfg["explore_steps"]):
            if t >= horizon:
                break
            if rng.random() > cfg["repeat_prob"]:
                a = action_space.sample()
            o, term = adapter.step(a)
            t += 1
            total_steps += 1
            f = flat(o)
            res.add(f)
            archive.update(cellmap(f), f, t, term, adapter.snapshot)
            if term:
                break
        history.append((it, archive.n, total_steps))
        if log_every and (it + 1) % log_every == 0:
            print(f"    iter {it + 1:6d}  cells {archive.n:7d}  steps {total_steps:8d}  "
                  f"{time.time() - t0:6.1f}s")
    return archive, res.data(), np.array(history)


# ============================== comparison =================================
def compare(reached, uniform):
    """Bounding-box coverage: a uniform sample counts as 'reachable-looking' if it
    lies within the [min, max] range of reached observations in every dim."""
    lo, hi = reached.min(0), reached.max(0)
    in_dim = (uniform >= lo) & (uniform <= hi)          # (N, d) per-dim check
    per_dim = in_dim.mean(0)
    worst = np.argsort(per_dim)[:3]
    return dict(
        frac_uniform_in_reached_bbox=float(in_dim.all(1).mean()),
        per_dim_frac_uniform_in_reached_range=per_dim.round(4).tolist(),
        worst_dims=[(int(d), float(per_dim[d])) for d in worst],
        reached_min=lo.tolist(),
        reached_max=hi.tolist(),
        space_low=None,   # filled in by caller
        space_high=None,
    )


def make_plots(out_dir, env_id, history, reached, uniform, pairs, rng):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(history[:, 2], history[:, 1])
    ax.set_xlabel("env steps")
    ax.set_ylabel("cells in archive")
    ax.set_title(f"{env_id}: archive growth")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "saturation.png"), dpi=120)
    plt.close(fig)

    d = reached.shape[1]
    pairs = [(i, j) for i, j in pairs if i < d and j < d]
    if not pairs:
        return
    sub = reached[rng.permutation(len(reached))[:20_000]]
    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4.5), squeeze=False)
    for ax, (i, j) in zip(axes[0], pairs):
        lo = sub[:, [i, j]].min(0)
        hi = sub[:, [i, j]].max(0)
        pad = 0.5 * (hi - lo) + 1e-6
        wlo, whi = lo - pad, hi + pad
        inwin = np.all((uniform[:, [i, j]] >= wlo) & (uniform[:, [i, j]] <= whi), axis=1)
        ax.scatter(uniform[inwin, i], uniform[inwin, j], s=3, alpha=0.3, c="tab:red",
                   label=f"space.sample() ({inwin.mean():.1%} in view)")
        ax.scatter(sub[:, i], sub[:, j], s=2, alpha=0.3, c="tab:blue", label="reached")
        bi, bj = reached[:, [i, j]].min(0), reached[:, [i, j]].max(0)
        ax.add_patch(plt.Rectangle((bi[0], bi[1]), bj[0] - bi[0], bj[1] - bi[1], fill=False,
                                   ec="black", lw=1.2, ls="--", label="reached bbox"))
        ax.set_xlim(wlo[0], whi[0])
        ax.set_ylim(wlo[1], whi[1])
        ax.set_xlabel(f"obs[{i}]")
        ax.set_ylabel(f"obs[{j}]")
        ax.legend(loc="upper right", fontsize=7, markerscale=4)
    fig.suptitle(env_id)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "scatter.png"), dpi=120)
    plt.close(fig)


# ============================ phase 2: state bank ===========================
def current_obs(adapter, flat):
    """Observation of the state the adapter is currently in, where the env exposes it."""
    u = adapter.u
    if hasattr(u, "_get_obs"):
        return flat(u._get_obs())
    if adapter.name == "classic" and hasattr(u, "state"):
        return flat(np.asarray(u.state, np.float32))
    return None


def collect_bank(adapter, flat, cellmap, archive, action_space, horizon, cfg, rng,
                 bank_dir, info):
    """Montezuma-style collection: for every (or a random subset of) archived cell, store the
    cell's own observation and observations recorded at random steps of short random rollouts
    started from it, together with the simulator state behind each observation."""
    

    ids = np.arange(archive.n)
    if BANK['max_cells'] and archive.n > BANK['max_cells']:
        ids = np.sort(rng.choice(archive.n, BANK['max_cells'], replace=False))
    obs_dim = len(archive.rep_obs[0])
    writer = BankWriter(bank_dir, adapter.name, len(ids) * BANK['per_cell'], obs_dim, info,
                        save_states=BANK['save_sim_states'])
    region_ids = {}

    def region_of(f):
        key = tuple((np.asarray(cellmap(f)) // BANK['region_factor']).tolist())
        return region_ids.setdefault(key, len(region_ids))

    t0 = time.time()
    for done, cid in enumerate(ids):
        hashes = set()

        def record(f, snap_fn, t, k, term):
            h = hash(np.asarray(f, np.float32).tobytes())
            if h in hashes:  # e.g. standing still produces identical observations
                return 0
            hashes.add(h)
            writer.add(f, snap_fn() if writer.save_states else None, cell_id=int(cid),
                       region=region_of(f), steps_from_start=int(t), steps_from_cell=int(k),
                       terminal=bool(term))
            return 1

        t_cell, term_cell = int(archive.traj_len[cid]), bool(archive.terminal[cid])
        # 1) the cell's own observation (re-read from the restored state if the observation
        #    space changed since the archive was written)
        if archive.rep_obs is not None:
            f0 = archive.rep_obs[cid]
        else:
            adapter.restore(archive.snaps[cid])
            f0 = current_obs(adapter, flat)
        got = record(f0, lambda: archive.snaps[cid], t_cell, 0, term_cell) if f0 is not None else 0

        # 2) short random rollouts from the cell, recording at random time steps
        for _ in range(BANK['max_attempts']):
            L = min(BANK['rollout_len'], horizon - t_cell)
            if got >= BANK['per_cell'] or term_cell or L <= 0:
                break
            adapter.restore(archive.snaps[cid])
            record_at = set(rng.choice(np.arange(1, L + 1), size=min(BANK['per_rollout'], L),
                                       replace=False).tolist())
            a = action_space.sample()
            for k in range(1, L + 1):
                if rng.random() > cfg["repeat_prob"]:
                    a = action_space.sample()
                o, term = adapter.step(a)
                if k in record_at:
                    got += record(flat(o), adapter.snapshot, t_cell + k, k, term)
                if term or got >= BANK['per_cell']:
                    break
        if (done + 1) % 2000 == 0:
            print(f"    bank: cells {done + 1}/{len(ids)} | observations {writer.n} | "
                  f"{time.time() - t0:5.1f}s")
    n = writer.close()
    print(f"    bank: {n} observations from {len(ids)} cells in {len(region_ids)} regions -> {bank_dir}/")
    return n


# =============================== env setup =================================
def setup(env_id, cfg):
    """Create the env and its adapter, and the flatten helper."""
    env = gym.make(env_id, **cfg.get("kwargs", {}))
    obs_space = env.unwrapped.observation_space
    if env.observation_space != obs_space:
        print("    NOTE: wrappers change the observation space; exploring the unwrapped env")
    flat_space = gym.spaces.flatten_space(obs_space)

    def flat(o):
        return np.asarray(gym.spaces.flatten(obs_space, o), dtype=np.float64)

    action_space = env.action_space
    action_space.seed(SEED)
    obs_space.seed(SEED)
    horizon = env.spec.max_episode_steps if env.spec and env.spec.max_episode_steps else 1000
    adapter = make_adapter(env, cfg["adapter"], flat, action_space)
    print(f"    adapter={adapter.name} horizon={horizon} obs_dim={flat_space.shape[0]} "
          f"iters={cfg['iters']} bins={cfg['bins']} cell_dims={cfg['cell_dims']}")
    return SimpleNamespace(env=env, obs_space=obs_space, flat_space=flat_space, flat=flat,
                           action_space=action_space, horizon=horizon, adapter=adapter)


def mujoco_sizes(adapter):
    if adapter.name != "mujoco":
        return None
    m = adapter.u.model
    return [int(m.nq), int(m.nv), int(m.na)]


def save_archive(path, archive, cellmap, cfg, S, env_id, metrics):
    from utils.state_bank import encode_snapshot
    n, kind = archive.n, S.adapter.name
    enc = [encode_snapshot(kind, sn) for sn in archive.snaps[:n]]
    data = dict(env_id=env_id, kind=kind, n=n,
                snaps=np.stack(enc) if kind in ("classic", "mujoco") else enc,
                rep_obs=np.asarray(archive.rep_obs, np.float64),
                traj_len=archive.traj_len[:n], terminal=archive.terminal[:n],
                cell_lo=cellmap.lo, cell_width=cellmap.width, cell_dims=cellmap.dims,
                obs_dim=int(S.flat_space.shape[0]), horizon=int(S.horizon), cfg=cfg,
                mujoco_sizes=mujoco_sizes(S.adapter),
                reached_min=metrics["reached_min"], reached_max=metrics["reached_max"])
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"    archive: {n} cells -> {path} ({os.path.getsize(path) / 1e6:.0f} MB)")


def rebuild_bank(env_id, args):
    """--bank-only: rebuild <out>/<env>/bank/ from archive.pkl, without exploring again."""
    
    out_dir = os.path.join(args.out, env_id.replace("/", "_"))
    path = os.path.join(out_dir, "archive.pkl")
    if not os.path.exists(path):
        print(f"    no archive at {path}; run without --bank-only first")
        return None
    with open(path, "rb") as f:
        data = pickle.load(f)

    cfg, matched = env_config(env_id)
    cfg["seed"] = SEED
    rng = np.random.default_rng(SEED)
    print(f"    config: {matched} | archive with {data['n']} cells")
    S = setup(env_id, cfg)
    adapter = ADAPTERS[data["kind"]](S.env)  # the archive's snapshots dictate the adapter
    archive = SimpleNamespace(
        n=data["n"], traj_len=data["traj_len"], terminal=data["terminal"],
        snaps=[decode_snapshot(data["kind"], sn, data["mujoco_sizes"]) for sn in data["snaps"]],
        rep_obs=data["rep_obs"])

    d_new, d_old = int(S.flat_space.shape[0]), int(data["obs_dim"])
    if d_new == d_old:
        cellmap = CellMapper.__new__(CellMapper)
        cellmap.lo, cellmap.width, cellmap.dims = data["cell_lo"], data["cell_width"], data["cell_dims"]
        reached_min, reached_max = data["reached_min"], data["reached_max"]
    else:  # e.g. Ant with include_cfrc_ext_in_observation switched off
        print(f"    observation dim changed {d_old} -> {d_new}: rebuilding the cell grid and "
              f"re-reading every cell's observation")
        warm = warmup(adapter, S.flat, S.action_space, cfg["warmup_episodes"], S.horizon, SEED)
        cellmap = CellMapper(S.flat_space, warm, cfg["cell_dims"], cfg["bins"])
        archive.rep_obs = None
        reached_min, reached_max = [np.inf] * d_new, [-np.inf] * d_new

    info = dict(env_id=env_id, adapter=adapter.name, horizon=int(S.horizon),
                obs_shape=list(S.obs_space.shape) if isinstance(S.obs_space, gym.spaces.Box) else [d_new],
                obs_dim=d_new, cell_dims=cfg["cell_dims"], bins=cfg["bins"],
                region_factor=BANK["region_factor"], seed=SEED,
                reached_min=list(reached_min), reached_max=list(reached_max))
    if adapter.name == "mujoco":
        info["mujoco_sizes"] = mujoco_sizes(adapter)
    n = collect_bank(adapter, S.flat, cellmap, archive, S.action_space, S.horizon, cfg, rng,
                     os.path.join(out_dir, "bank"), info)
    S.env.close()
    return dict(env_id=env_id, adapter=adapter.name, cells=int(archive.n), env_steps=0,
                bank_observations=n, frac_uniform_in_reached_bbox=float("nan"), worst_dims=[])


# ================================= main ====================================
def run_env(env_id, args):
    cfg, matched = env_config(env_id)
    if args.iters is not None:
        cfg["iters"] = args.iters
    cfg["seed"] = SEED
    rng = np.random.default_rng(SEED)
    print(f"    config: {matched}")

    S = setup(env_id, cfg)
    env, obs_space, flat_space, flat = S.env, S.obs_space, S.flat_space, S.flat
    action_space, horizon, adapter = S.action_space, S.horizon, S.adapter

    warm = warmup(adapter, flat, action_space, cfg["warmup_episodes"], horizon, SEED)
    cellmap = CellMapper(flat_space, warm, cfg["cell_dims"], cfg["bins"])

    t0 = time.time()
    archive, reached, history = go_explore(adapter, flat, cellmap, action_space, cfg,
                                           horizon, rng, MAX_REACHED)
    reached = np.concatenate([reached, warm])
    explore_time = time.time() - t0

    uniform = np.stack([flat(obs_space.sample()) for _ in range(N_UNIFORM)])
    metrics = compare(reached, uniform)
    fs_lo, fs_hi = flat_space.low.astype(np.float64), flat_space.high.astype(np.float64)
    metrics["space_low"] = [None if not np.isfinite(v) else float(v) for v in fs_lo]
    metrics["space_high"] = [None if not np.isfinite(v) else float(v) for v in fs_hi]
    metrics.update(env_id=env_id, adapter=adapter.name, cells=int(archive.n),
                   env_steps=int(history[-1, 2]) if len(history) else 0,
                   explore_seconds=round(explore_time, 1), config={k: cfg[k] for k in cfg if k != "plot_pairs"},
                   max_traj_len_in_archive=int(archive.traj_len[: archive.n].max()))

    out_dir = os.path.join(args.out, env_id.replace("/", "_"))
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "reachable.npz"),
                        reached=reached.astype(np.float32),
                        cell_representatives=np.array(archive.rep_obs, dtype=np.float32),
                        cell_traj_len=archive.traj_len[: archive.n],
                        cell_terminal=archive.terminal[: archive.n],
                        uniform=uniform, history=history)
    make_plots(out_dir, env_id, history, reached, uniform, cfg["plot_pairs"], rng)
    if SAVE_ARCHIVE:
        save_archive(os.path.join(out_dir, "archive.pkl"), archive, cellmap, cfg, S, env_id, metrics)
    # state bank for later sampling (always written)
    info = dict(env_id=env_id, adapter=adapter.name, horizon=int(horizon),
                obs_shape=list(obs_space.shape) if isinstance(obs_space, gym.spaces.Box)
                else [int(flat_space.shape[0])],
                obs_dim=int(flat_space.shape[0]), cell_dims=cfg["cell_dims"], bins=cfg["bins"],
                region_factor=BANK['region_factor'], seed=SEED,
                reached_min=metrics["reached_min"], reached_max=metrics["reached_max"])
    if adapter.name == "mujoco":
        info["mujoco_sizes"] = mujoco_sizes(adapter)
    metrics["bank_observations"] = collect_bank(
        adapter, flat, cellmap, archive, action_space, horizon, cfg, rng,
        os.path.join(out_dir, "bank"), info)
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    env.close()
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--envs", nargs="+", default=ENVS,
                   help="gym env ids (default: all envs in ENVS at the top of this file)")
    p.add_argument("--import", dest="import_module", nargs="*", default=[],
                   help="modules to import first (the ones registering custom envs)")
    p.add_argument("--iters", type=int, default=None, help="Go-Explore iterations (default 10000)")
    p.add_argument("--out", default="reachability_results", help="output directory")
    p.add_argument("--bank-only", action="store_true",
                   help="rebuild the bank from a previous run's archive.pkl (no exploration)")
    args = p.parse_args()

    import sys
    sys.path.insert(0, os.getcwd())
    for m in IMPORTS + args.import_module:
        importlib.import_module(m)

    rows = []
    for env_id in args.envs:
        print(f"\n=== {env_id} ===")
        try:
            m = rebuild_bank(env_id, args) if args.bank_only else run_env(env_id, args)
            if m is None:
                continue
        except (gym.error.Error, ModuleNotFoundError) as e:
            print(f"    skipped: {e}")
            continue
        rows.append(m)

    print("\n" + "=" * 96)
    print(f"{'env':24s}{'adapter':10s}{'cells':>8s}{'steps':>10s}{'in bbox':>10s}   "
          f"lowest per-dim in-range fractions")
    print("-" * 96)
    for m in rows:
        worst = ", ".join(f"obs[{d}]={f:.1%}" for d, f in m["worst_dims"])
        print(f"{m['env_id']:24s}{m['adapter']:10s}{m['cells']:8d}{m['env_steps']:10d}"
              f"{m['frac_uniform_in_reached_bbox']:10.2%}   {worst}")
    print("\n'in bbox' = fraction of observation_space.sample() draws inside the reached "
          "[min, max] range in every dimension.")
    print(f"Per-env data, metrics.json and plots in ./{args.out}/")


if __name__ == "__main__":
    main()
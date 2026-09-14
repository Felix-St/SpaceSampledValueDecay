"""RiverSwim: a classic hard-exploration tabular benchmark.

Reference
---------
A. L. Strehl and M. L. Littman, "An analysis of model-based Interval Estimation
for Markov Decision Processes", JCSS 2008.

The agent is a swimmer in a river flowing left. Swimming LEFT (with the current)
always succeeds and yields a small reward at the leftmost state. Swimming RIGHT
(against the current) often fails, but the rightmost state pays a reward 200x
larger. 
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from gymnasium.envs.registration import register
import gymnasium as gym
from gymnasium import spaces

register(
    id="RiverSwim-v0",
    entry_point="environments.river_swim:RiverSwimEnv",  # module:path_to_class
    kwargs={"n_states": 6, "horizon": 200},
)

LEFT, RIGHT = 0, 1
ACTION_NAMES = {LEFT: "left", RIGHT: "right"}


class OneHot(spaces.Box):
    def __init__(self, n: int, seed=None):
        self.n = n
        super().__init__(low=0.0, high=1.0, shape=(n,), dtype=np.float32, seed=seed)

    def sample(self, mask=None):
        i = self.np_random.integers(self.n)
        x = np.zeros(self.n, dtype=np.float32)
        x[i] = 1.0
        return x

    def contains(self, x):
        x = np.asarray(x)
        return (
            x.shape == (self.n,)
            and np.isin(x, (0.0, 1.0)).all()
            and x.sum() == 1.0
        )

    def __repr__(self):
        return f"OneHot({self.n})"
    

class RiverSwimEnv(gym.Env):
    """RiverSwim as a Gymnasium environment.

    Parameters
    ----------
    n_states:
        Length of the chain (>= 2). The canonical benchmark uses 6; larger
        values make exploration exponentially harder for dithering agents.
    horizon:
        Steps before the episode is truncated. RiverSwim is natively an
        infinite-horizon average-reward problem, so this is only a bookkeeping
        device: `truncated=True` is returned (never `terminated=True`), which
        keeps bootstrapping correct in value-based agents. Use ``None`` for a
        never-ending episode.
    small_reward, large_reward:
        Payoffs at the left and right ends of the chain.
    reward_on_transition:
        If ``False`` (default) the reward depends on ``(s, a)`` and is paid
        whenever RIGHT is chosen in the rightmost state. If ``True`` it depends
        on ``(s, a, s')`` and is paid only when the swimmer actually holds
        position at the far right. Both conventions appear in the literature;
        the second is slightly harder.
    obs_mode:
        ``"discrete"`` -> ``Discrete(n_states)`` (tabular agents).
        ``"one_hot"``  -> ``Box(0, 1, (n_states,), float32)`` (function approx).
        ``"scalar"``   -> ``Box(0, 1, (1,), float32)``, position normalised to
        [0, 1] (useful when you want a genuinely low-dimensional input).
    start_state:
        Fixed starting index, or ``None`` to sample uniformly from the two
        leftmost states as in the original paper.

    Attributes
    ----------
    P : np.ndarray, shape (S, A, S)   -- transition kernel
    R : np.ndarray, shape (S, A)      -- expected reward
    R_sas : np.ndarray, shape (S, A, S) -- per-transition reward
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 8}

    def __init__(
        self,
        n_states: int = 6,
        horizon: Optional[int] = 200,
        small_reward: float = 5,
        large_reward: float = 10000,
        reward_on_transition: bool = False,
        obs_mode: str = "one_hot",
        start_state: Optional[int] = 0,
        render_mode: Optional[str] = None,
    ) -> None:

        if n_states < 2:
            raise ValueError("n_states must be >= 2")
        if obs_mode not in ("discrete", "one_hot", "scalar"):
            raise ValueError(f"unknown obs_mode {obs_mode!r}")
        if start_state is not None and not 0 <= start_state < n_states:
            raise ValueError("start_state out of range")

        self.n_states = int(n_states)
        self.horizon = horizon
        self.small_reward = float(small_reward)
        self.large_reward = float(large_reward)
        self.reward_on_transition = bool(reward_on_transition)
        self.obs_mode = obs_mode
        self.start_state = start_state
        self.render_mode = render_mode

        # ---- spaces -------------------------------------------------------
        self.action_space = spaces.Discrete(2)
        if obs_mode == "discrete":
            self.observation_space = spaces.Discrete(self.n_states)
        elif obs_mode == "one_hot":
            self.observation_space = OneHot(self.n_states)
        else:  # scalar
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32
            )

        # ---- model --------------------------------------------------------
        self.P, self.R_sas = self._build_model()
        self.R = np.einsum("sap,sap->sa", self.P, self.R_sas)

        self._state = 0
        self._elapsed = 0

    # ----------------------------------------------------------------- model
    def _build_model(self) -> tuple[np.ndarray, np.ndarray]:
        S = self.n_states
        P = np.zeros((S, 2, S), dtype=np.float64)
        R = np.zeros((S, 2, S), dtype=np.float64)

        for s in range(S):
            # LEFT: deterministic drift with the current.
            P[s, LEFT, max(s - 1, 0)] = 1.0

            # RIGHT: stochastic struggle against the current.
            if s == 0:
                P[s, RIGHT, 1] = 0.40
                P[s, RIGHT, 0] = 0.60
            elif s == S - 1:
                P[s, RIGHT, s] = 0.60
                P[s, RIGHT, s - 1] = 0.40
            else:
                P[s, RIGHT, s + 1] = 0.35
                P[s, RIGHT, s] = 0.60
                P[s, RIGHT, s - 1] = 0.05

        if self.reward_on_transition:
            R[0, LEFT, 0] = self.small_reward
            R[S - 1, RIGHT, S - 1] = self.large_reward
        else:
            R[0, LEFT, :] = self.small_reward
            R[S - 1, RIGHT, :] = self.large_reward

        assert np.allclose(P.sum(axis=-1), 1.0)
        return P, R

    # ------------------------------------------------------------ gym API
    def _obs(self) -> Any:
        if self.obs_mode == "discrete":
            return int(self._state)
        if self.obs_mode == "one_hot":
            v = np.zeros(self.n_states, dtype=np.float32)
            v[self._state] = 1.0
            return v
        return np.array([self._state / (self.n_states - 1)], dtype=np.float32)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[Any, dict]:
        super().reset(seed=seed)
        if self.start_state is None:
            self._state = int(self.np_random.integers(0, 2))
        else:
            self._state = int(self.start_state)
        self._elapsed = 0
        if self.render_mode == "human":
            self.render()
        return self._obs(), {"state": self._state}

    def step(self, action: int) -> tuple[Any, float, bool, bool, dict]:
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action!r}")

        s = self._state
        s_next = int(self.np_random.choice(self.n_states, p=self.P[s, action]))
        reward = float(self.R_sas[s, action, s_next])

        self._state = s_next
        self._elapsed += 1

        terminated = False  # continuing task: never absorbing
        truncated = self.horizon is not None and self._elapsed >= self.horizon

        info = {"state": s_next, "prev_state": s, "action": ACTION_NAMES[action]}
        if self.render_mode == "human":
            self.render()
        return self._obs(), reward, terminated, truncated, info

"""DeepSea exploration benchmark as a Gymnasium environment (Stable-Baselines3 compatible).

Follows the dynamics of bsuite's `deep_sea` (Osband et al., "Behaviour Suite for RL", 2020):

- N x N grid. The agent starts in the top-left cell (row 0, col 0).
- Every step the agent descends one row, so an episode always lasts exactly N steps.
- Two actions: "right" (col + 1) and "left" (col - 1, clipped at 0).
- Which action index means "right" is randomised per cell (fixed by `mapping_seed`),
  so a policy like "always press 1" does not solve the task.
- Moving right costs `move_cost / N`. Reaching the bottom-right corner and pressing
  right there gives +1. The only way to get it is to go right on all N steps, so
  the optimal return is 1 - move_cost and a uniformly random policy finds the
  treasure with probability 2^-N.

Stochastic variant (`deterministic=False`, bsuite's deep_sea_stochastic): a "right"
move fails with probability 1/N, and N(0, 1) reward noise is added at the two
bottom corners.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environments.river_swim import OneHot


class DeepSeaEnv(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(
        self,
        size: int = 10,
        deterministic: bool = True,
        randomize_actions: bool = True,
        move_cost: float = 0.01,
        mapping_seed: int = 42,
        obs_type: str = "grid",
        render_mode: str | None = None,
    ):
        super().__init__()
        if size < 2:
            raise ValueError("size must be >= 2")
        if obs_type not in ("grid", "coords"):
            raise ValueError("obs_type must be 'grid' or 'coords'")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode {render_mode!r}")

        self.size = size
        self.deterministic = deterministic
        self.randomize_actions = randomize_actions
        self.move_cost = move_cost
        self.obs_type = obs_type
        self.render_mode = render_mode

        # The action mapping is part of the MDP, so it is drawn from its own seed and
        # does NOT change when reset(seed=...) is called. Entry [r, c] is the action
        # index that moves right in cell (r, c).
        if randomize_actions:
            rng = np.random.default_rng(mapping_seed)
            self._action_mapping = rng.integers(0, 2, size=(size, size), dtype=np.int64)
        else:
            self._action_mapping = np.ones((size, size), dtype=np.int64)

        self.action_space = spaces.Discrete(2)
        if obs_type == "grid":
            # One-hot N x N image; all zeros after the final step (as in bsuite).
            # SB3's MlpPolicy flattens this automatically.
            self.observation_space = OneHot(size**2)
        else:
            # (row, col) scaled to [0, 1].
            self.observation_space = spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32)

        self._row = 0
        self._col = 0
        self._denoised_return = 0.0

    # ------------------------------------------------------------------ helpers
    def right_action(self, row: int | None = None, col: int | None = None) -> int:
        """Action index that moves right in (row, col); defaults to the current cell.
        Intended for tests and oracle baselines, not for the agent."""
        row = self._row if row is None else row
        col = self._col if col is None else col
        return int(self._action_mapping[row, col])

    def _get_obs(self) -> np.ndarray:
        if self.obs_type == "grid":
            obs = np.zeros(self.size**2, dtype=np.float32)
            if self._row < self.size:
                obs[self._row * self.size + self._col] = 1.0
            return obs
        return np.array(
            [self._row / self.size, self._col / max(self.size - 1, 1)], dtype=np.float32
        )

    # ---------------------------------------------------------------- gym API
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(
            seed=seed
        )  # seeds self.np_random (transition / reward noise only)
        self._row = 0
        self._col = 0
        self._denoised_return = 0.0
        return self._get_obs(), {}

    def step(self, action):
        if self._row >= self.size:
            raise RuntimeError("Episode is over; call reset().")

        action = int(action)
        n = self.size
        row, col = self._row, self._col
        action_right = action == self._action_mapping[row, col]

        reward = 0.0
        found_treasure = False

        # Treasure: pressing right while in the last column (only reachable on the last row).
        if col == n - 1 and action_right:
            reward += 1.0
            self._denoised_return += 1.0
            found_treasure = True

        # Reward noise at the bottom corners in the stochastic version.
        if not self.deterministic and row == n - 1 and col in (0, n - 1):
            reward += float(self.np_random.standard_normal())

        # Transitions.
        if action_right:
            if self.deterministic or self.np_random.random() > 1.0 / n:
                self._col = min(col + 1, n - 1)
            reward -= self.move_cost / n
            self._denoised_return -= self.move_cost / n
        else:
            self._col = max(col - 1, 0)
        self._row += 1

        terminated = self._row == n
        info = {
            "treasure": found_treasure,
            "denoised_return": self._denoised_return,
        }
        if self.render_mode == "ansi":
            info["render"] = self.render()
        return self._get_obs(), float(reward), terminated, False, info

    def render(self):
        if self.render_mode != "ansi":
            return None
        lines = []
        for r in range(self.size):
            lines.append(
                " ".join(
                    (
                        "A"
                        if (r == self._row and c == self._col)
                        else (
                            "$" if (r == self.size - 1 and c == self.size - 1) else "."
                        )
                    )
                    for c in range(self.size)
                )
            )
        return "\n".join(lines)


def register_deep_sea() -> None:
    """Register `DeepSea-v0` so you can call gym.make("DeepSea-v0", size=12)."""
    if "DeepSea-v0" not in gym.registry:
        gym.register(id="DeepSea-v0", entry_point=DeepSeaEnv)


register_deep_sea()

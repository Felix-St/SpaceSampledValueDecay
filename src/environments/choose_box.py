from __future__ import annotations

from minigrid.core.grid import Grid
from minigrid.core.world_object import Box, Wall
from minigrid.core.mission import MissionSpace
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.actions import Actions
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX

import numpy as np
import time

from gymnasium.envs.registration import register

register(
    id="ChooseBoxEnv-v0",
    entry_point="environments.choose_box:ChooseBoxEnv",  # module:path_to_class
    kwargs={
        "size": 5,
        "max_steps": 200,
        "drift_probability": None,
        "drift_timesteps": None,
    },
)


class ChooseBoxEnv(MiniGridEnv):
    """
    Two-box selection task with non-stationary reward.

    - Spawns two boxes with different colors.
    - Mission: pick up the target-colored box.
    - Episode terminates immediately on picking either box
      (success if target, else failure).

    Drift
    -----
    The identity of the rewarding color can flip *during* an episode. A drift is
    drawn once per environment step from a dedicated RNG (`drift_rng`), so the
    set of drift timesteps is a deterministic function of
    (`seed`, `drift_probability`, `min_steps_between_drifts`) and is *identical
    across algorithms, episode lengths and reset patterns*. That makes it valid
    to plot return against the global timestep and overlay the change points.

    `total_time` is the global step counter: it is incremented on every `step()`
    call and is never reset by `reset()`.
    """

    def __init__(
        self,
        size,
        drift_probability,
        drift_timesteps,
        max_steps: int | None = None,
        seed=0,
        min_steps_between_drifts: int = 0,
        reveal_target_in_mission: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        drift_probability:
            Per-*timestep* probability of a drift (previously per-episode). With
            episodes of length L, an old per-episode value p is roughly
            equivalent to p / L here.
        min_steps_between_drifts:
            Minimum number of global steps between two drifts. 0 disables the
            constraint. Useful to stop drifts clustering into a single burst.
        reveal_target_in_mission:
            If True the mission string names the current target color, which
            makes a mid-episode drift directly observable to any agent that
            reads the mission. Default False keeps the mission constant, so the
            drift is latent and must be inferred from reward.
        """
        # Keep these for default agent placement; can be overridden in _gen_grid
        self.agent_start_pos = None
        self.agent_start_dir = 0

        self.reward_sign = 1
        self.reward_flip_prob = drift_probability
        self.drift_timesteps = drift_timesteps
        self.min_steps_between_drifts = int(min_steps_between_drifts)
        self.reveal_target_in_mission = reveal_target_in_mission

        # Global step counter, shared across episodes. Never reset.
        self.total_time = 0
        self.last_drift_time = -np.inf

        self.drift_rng = np.random.default_rng(seed)
        self.change_points: list[int] = []

        self.target_color = "green"
        self.distractor_color = "red"

        # Keep object references for identity comparison at pick-up time.
        # Set before super().__init__() in case it triggers a grid generation.
        self.box_target: Box | None = None
        self.box_other: Box | None = None

        if reveal_target_in_mission:
            mission_space = MissionSpace(
                mission_func=lambda color: f"pick up the {color} box",
                ordered_placeholders=[self._available_colors()],
            )
        else:
            mission_space = MissionSpace(mission_func=lambda: "pick up the target box")

        if max_steps is None:
            max_steps = 4 * size * size

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _available_colors():
        return ["red", "green", "blue", "yellow", "purple", "grey"]

    # ------------------------------------------------------------------ drift

    def _make_mission(self) -> str:
        if self.reveal_target_in_mission:
            return f"pick up the {self.target_color} box"
        return "pick up the target box"

    def _apply_drift(self) -> None:
        """Swap which color is rewarding. Does not touch the visible grid."""
        self.reward_sign = -1 * self.reward_sign

        self.target_color, self.distractor_color = (
            self.distractor_color,
            self.target_color,
        )

        # The box that was the distractor now carries the rewarding color, so
        # the identity references have to follow the color swap.
        if self.box_target is not None or self.box_other is not None:
            self.box_target, self.box_other = self.box_other, self.box_target

        self.change_points.append(self.total_time)
        self.last_drift_time = self.total_time

        self.mission = self._make_mission()

    def _maybe_drift(self) -> bool:
        """
        Draw exactly one random number per global step, whatever else happens,
        so the drift stream stays aligned with `total_time`.
        """

        if self.reward_flip_prob is not None:
            u = float(self.drift_rng.random())

            if self.reward_flip_prob <= 0.0:
                return False
            if self.total_time - self.last_drift_time < self.min_steps_between_drifts:
                return False
            if u >= self.reward_flip_prob:
                return False
        elif self.drift_timesteps is not None:
            if self.total_time in self.drift_timesteps:
                self._apply_drift()
                return True
            else:
                return False
        else:
            raise Exception("Unknown kind of drift")

        self._apply_drift()
        return True

    # ------------------------------------------------------------------- env

    def _gen_grid(self, width: int, height: int):
        # Create empty grid and enclosing walls
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # NOTE: no drift is drawn here any more. Drift is a function of the
        # global timestep only, so that different algorithms with different
        # episode lengths see the exact same change points.

        # Place agent
        self.place_agent()

        # Place the target and distractor boxes at random free cells
        self.box_target = Box(self.target_color)
        self.place_obj(self.box_target)

        self.box_other = Box(self.distractor_color)
        self.place_obj(self.box_other)

        # Set mission text
        self.mission = self._make_mission()

    def sample_random_state(self, rng=None):
        cells = [
            (x, y) for x in range(1, self.width - 1) for y in range(1, self.height - 1)
        ]
        agent, tgt, dst = (cells[i] for i in rng.choice(len(cells), 3, replace=False))
        agent_dir = int(rng.integers(4))

        grid = Grid(self.width, self.height)
        grid.wall_rect(0, 0, self.width, self.height)
        grid.set(*tgt, Box(self.target_color))
        grid.set(*dst, Box(self.distractor_color))

        n = self.agent_view_size
        top_x = agent[0] - (
            n // 2 if agent_dir % 2 else (n - 1 if agent_dir == 2 else 0)
        )
        top_y = agent[1] - (
            n // 2 if agent_dir % 2 == 0 else (n - 1 if agent_dir == 3 else 0)
        )
        view = grid.slice(top_x, top_y, n, n)
        for _ in range(agent_dir + 1):
            view = view.rotate_left()

        img = view.encode()

        n_obj, n_col = len(OBJECT_TO_IDX), len(COLOR_TO_IDX)
        out = np.zeros(
            (*img.shape[:2], n_obj + n_col + len(STATE_TO_IDX)), dtype=np.uint8
        )
        i, j = np.indices(img.shape[:2])
        out[i, j, img[..., 0]] = 1
        out[i, j, n_obj + img[..., 1]] = 1
        out[i, j, n_obj + n_col + img[..., 2]] = 1
        return out

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        info.update(self._drift_info(drifted=False))
        return obs, info

    def _drift_info(self, drifted: bool) -> dict:
        return {
            "global_step": self.total_time,
            "drift": drifted,
            "n_drifts": len(self.change_points),
            "reward_sign": self.reward_sign,
            "target_color": self.target_color,
            "change_points": tuple(self.change_points),
        }

    def step(self, action: int):
        # Drift is resolved *before* the action is executed, so the reward
        # collected at global step t already follows the rule announced at t.
        drifted = self._maybe_drift()

        obs, reward, terminated, truncated, info = super().step(action)

        self.total_time = self.total_time + 1

        # Early return if already done
        if terminated or truncated:
            info.update(self._drift_info(drifted))
            return obs, reward, terminated, truncated, info

        # If the agent just picked something up, check which box it is
        if action == Actions.pickup:
            if self.carrying is not None:
                # Identity check is robust to how colors are represented internally
                if self.carrying is self.box_target:
                    # Shaped success reward consistent with many MiniGrid tasks
                    reward = 10.0 - 9 * (self.step_count / self.max_steps)
                    terminated = True
                elif self.carrying is self.box_other:
                    reward = 10.0 - 9 * (self.step_count / self.max_steps)
                    reward = -1 * reward
                    terminated = True

        info.update(self._drift_info(drifted))
        return obs, reward, terminated, truncated, info

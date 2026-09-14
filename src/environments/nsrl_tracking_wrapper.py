"""
    Simple wrappers to keep track of absolut times of drifts and small compatibility 
    adaption (observation_space) to allow sampling from the observation space in 
    the modified SB3 algorithms.
"""

from ns_gym.wrappers import NSClassicControlWrapper, MujocoWrapper
from typing import Any, Union

class LifelongNSWrapperSB3ClassicControl(NSClassicControlWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = self.observation_space["state"]
        self.change_points = []
        self.total_time = 0
    
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """ 
            We only return the state for SB3 agents e.g. we do not assume knowledge about the time
            the drift occurs. 
        """
        obs, info = super().reset(seed=seed, options=options)
        return obs["state"], info

    def step(self, action: Union[float, int]):
        """Step through environment and update environmental parameters

        Args:
            action (Union[float,int]): Action to take in environment

        Returns:
            tuple[dict[str, Any], base.Reward, bool, bool, dict[str, Any]]: NS-Gym Observation dictionary, reward, done flag, truncated flag, info dictionary
        """
        self.total_time = self.total_time + 1
        obs, reward, terminated, truncated, info = super().step(action=action)

        change = False

        for param in obs["env_change"].keys():
            if obs["env_change"][param] == 1:
                change = True

        if change:
            self.change_points.append(self.total_time)

        return obs["state"], reward, terminated, truncated, info
    

class LifelongNSWrapperSB3Mujoco(MujocoWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = self.observation_space["state"]
        self.change_points = []
        self.total_time = 0


    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """ 
            We only return the state for SB3 agents e.g. we do not assume knowledge about the time
            the drift occurs.
        """
        obs, info = super().reset(seed=seed, options=options)
        return obs["state"], info

    def step(self, action: Union[float, int]):
        """Step through environment and update environmental parameters

        Args:
            action (Union[float,int]): Action to take in environment

        Returns:
            tuple[dict[str, Any], base.Reward, bool, bool, dict[str, Any]]: NS-Gym Observation dictionary, reward, done flag, truncated flag, info dictionary
        """
        self.total_time = self.total_time + 1
        obs, reward, terminated, truncated, info = super().step(action=action)

        change = False

        for param in obs["env_change"].keys():
            if obs["env_change"][param] == 1:
                change = True

        if change:
            self.change_points.append(self.total_time)

        return obs["state"], reward, terminated, truncated, info
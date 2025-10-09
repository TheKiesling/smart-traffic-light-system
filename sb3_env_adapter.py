# sb3_env_adapter.py
import numpy as np
import gymnasium as gym

from traffic_alpyne_env import TrafficAlpyneEnv  # tu core env ya probado

class SB3TrafficEnv(gym.Env):
    """
    Adaptador Gymnasium para usar TrafficAlpyneEnv con Stable-Baselines3.
    """
    metadata = {"render_modes": []}

    def __init__(self, model_path: str):
        super().__init__()
        self.core = TrafficAlpyneEnv(model_path)
        self.observation_space = self.core.observation_space
        self.action_space = self.core.action_space

    def reset(self, *, seed=None, options=None):
        obs, info = self.core.reset(seed=seed, options=options)
        return obs, info

    def step(self, action):
        obs, reward, done, trunc, info = self.core.step(action)
        # Gymnasium step API: (obs, reward, terminated, truncated, info)
        return obs, reward, bool(done), bool(trunc), info

    def close(self):
        self.core.close()

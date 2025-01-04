import gymnasium as gym
from typing import override

from .envrionment import Environment
from ..render_modes import RenderMode


class BasicEnvironment(Environment):
    
    def __init__(self, env_id: str, render_mode : RenderMode = RenderMode.NONE, seed : int = None):
        super().__init__(env_id, render_mode, seed)

    @override
    def step(self, action):
        return self.inner_env.step(action)

    @override
    def reset(self, seed: int = None):
        return self.inner_env.reset(seed=seed)

    @override
    def get_action_space(self):
        return self.inner_env.action_space

    @override
    def random_step(self):
        return self.inner_env.action_space.sample()

    @override
    def get_observation_space(self):
        return self.inner_env.observation_space

    @override
    def close(self):
        return self.inner_env.close()
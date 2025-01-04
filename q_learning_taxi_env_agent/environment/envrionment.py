import gymnasium as gym

from abc import ABC, abstractmethod

from ..render_modes import RenderMode


class Environment(ABC):

    def __init__(self, env_id: str, render_mode : RenderMode = RenderMode.NONE, seed : int = None):
        self.env_id = env_id
        self.render_mode = render_mode
        self.seed = seed
        self.inner_env = gym.make(env_id, render_mode=render_mode.value)

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def reset(self, seed: int = None):
        pass

    @abstractmethod
    def get_action_space(self):
        pass

    @abstractmethod
    def random_step(self):
        pass

    @abstractmethod
    def get_observation_space(self):
        pass

    @abstractmethod
    def close(self):
        pass
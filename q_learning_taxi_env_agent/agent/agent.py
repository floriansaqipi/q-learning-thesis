
from abc import ABC, abstractmethod

from .utils import ExperienceHandler
from ..environment import Environment


class Agent(ExperienceHandler, ABC):

    def __init__(self, env: Environment):
        self.env = env
        self.training_error = []

    @abstractmethod
    def get_action(self, obs):
        pass

    @abstractmethod
    def get_best_action(self, obs):
        pass

    @abstractmethod
    def update(self, obs, action, reward, terminated, next_obs, truncated = None):
        pass

    def end_of_episode_hook(self):
        pass

    def get_training_error(self):
        return self.training_error
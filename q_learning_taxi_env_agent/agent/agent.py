
from abc import ABC, abstractmethod

from ..environment import Environment


class Agent(ABC):

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
    def update(self, obs, action, reward, terminated, next_obs):
        pass

    def get_training_error(self):
        return self.training_error
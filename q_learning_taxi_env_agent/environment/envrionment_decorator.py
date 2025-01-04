from abc import ABC
from typing import override

from .basic_environment import Environment


class EnvironmentDecorator(Environment, ABC):
    def __init__(self, env: Environment):
        super().__init__(env.env_id, env.render_mode, env.seed)
        self.env = env

    @override
    def step(self, action):
        return self.env.step(action)

    @override
    def reset(self, seed: int = None):
        return self.env.reset()

    @override
    def get_action_space(self):
        return self.env.get_action_space()

    @override
    def random_step(self):
        return self.env.random_step()

    @override
    def get_observation_space(self):
        return self.env.get_observation_space()

    @override
    def close(self):
        return self.env.close()




import gymnasium as gym
from gymnasium.vector.async_vector_env import AsyncVectorEnv

from .environment_wrapper import EnvironmentWrapper
from .environment import Environment


class VectorizedEnvironment(EnvironmentWrapper):
    def __init__(self, env: Environment, number_of_envs: int = 1):
        super().__init__(env)

        inner_envs = [lambda : env.inner_env] + [lambda: self.make_gym_env() for _ in range(number_of_envs - 1)]
        self.inner_env = AsyncVectorEnv(inner_envs)

    def get_action_space(self):
        return self.inner_env.action_space[0]
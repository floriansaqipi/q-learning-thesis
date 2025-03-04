
import gymnasium as gym
import ale_py


from ..render_modes import RenderMode


class Environment:

    def __init__(self, env_id: str, render_mode: RenderMode = RenderMode.NONE, seed: int = None):
        gym.register_envs(ale_py)

        self.env_id = env_id
        self.render_mode = render_mode
        self.seed = seed

        self.inner_env = self.make_gym_env()

    def make_gym_env(self):
        return gym.make(self.env_id, render_mode=self.render_mode.value)

    def step(self, action):
        return self.inner_env.step(action)

    def reset(self, seed: int = None):
        return self.inner_env.reset(seed=seed)

    def get_action_space(self):
        return self.inner_env.action_space

    def random_step(self):
        return self.inner_env.action_space.sample()

    def get_observation_space(self):
        return self.inner_env.observation_space

    def close(self):
        return self.inner_env.close()

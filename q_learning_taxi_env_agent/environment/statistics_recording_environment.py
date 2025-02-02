import gymnasium as gym
from typing_extensions import override

from .envrionment import Environment
from .environment_wrapper import EnvironmentWrapper
from .episode_queue_provider import EpisodeQueueProvider


class StatisticsRecordingEnvironment(EnvironmentWrapper, EpisodeQueueProvider):
    def __init__(self, env: Environment, n_episodes: int):
        super().__init__(env)
        self.n_episodes = n_episodes
        self.inner_env = gym.wrappers.RecordEpisodeStatistics(self.env.inner_env, buffer_length=self.n_episodes)

    def get_return_queue(self):
        return self.inner_env.return_queue

    def get_length_queue(self):
        return self.inner_env.length_queue
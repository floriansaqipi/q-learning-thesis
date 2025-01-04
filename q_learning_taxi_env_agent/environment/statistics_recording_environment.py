import gymnasium as gym
from typing_extensions import override

from .envrionment import Environment
from .envrionment_decorator import EnvironmentDecorator
from .episode_queue_provider import EpisodeQueueProvider


class StatisticsRecordingEnvironment(EnvironmentDecorator, EpisodeQueueProvider):
    def __init__(self, env: Environment, n_episodes: int):
        super().__init__(env)
        self.n_episodes = n_episodes
        self.env.inner_env = gym.wrappers.RecordEpisodeStatistics(self.env.inner_env, buffer_length=self.n_episodes)

    @override
    def get_return_queue(self):
        return self.env.inner_env.return_queue

    @override
    def get_length_queue(self):
        return self.env.inner_env.length_queue
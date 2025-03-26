from abc import ABC, abstractmethod

from tqdm import tqdm

from ..agent.utils import ExperienceHandler
from ..environment import Environment
from ..agent import Agent


class Player(ExperienceHandler, ABC):
    def __init__(self, agent: Agent, env: Environment, n_episodes: int = None, printing_enabled: bool = False):

        self.agent = agent
        self.env = env
        self.n_episodes = n_episodes
        self.printing_enabled = printing_enabled
        if printing_enabled:
            self.progress_bar = tqdm(total=self.n_episodes)

    def save_progress(self, save_frequency=0, episode_number=0, return_queue=None, length_queue=None):
        self.agent.save_progress(save_frequency, episode_number, return_queue, length_queue)

    def load_progress(self):
        self.agent.load_progress()

    @abstractmethod
    def play(self):
        pass

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'progress_bar' in state:
            del state['progress_bar']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.printing_enabled:
            self.progress_bar = tqdm(total=self.n_episodes)

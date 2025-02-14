from abc import ABC, abstractmethod

from tqdm import tqdm

from ..agent.utils import ExperienceHandler
from ..environment import Environment
from ..agent import Agent


class Player(ExperienceHandler, ABC):
    def __init__(self, agent: Agent, env: Environment, n_episodes: int = None):
        self.agent = agent
        self.env = env
        self.n_episodes = n_episodes
        self.progress_bar = tqdm(total=self.n_episodes)

    def save_progress(self, file_name: str):
        self.agent.save_progress(file_name)

    def load_progress(self, file_name: str):
        self.agent.load_progress(file_name)

    @abstractmethod
    def play(self):
        pass

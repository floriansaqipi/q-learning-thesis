from abc import ABC, abstractmethod

from tqdm import tqdm

from ..environment import Environment


class Player(ABC):
    def __init__(self, env: Environment, n_episodes: int = None):
        self.env = env
        self.n_episodes = n_episodes
        self.progress_bar = tqdm(total=self.n_episodes)


    @abstractmethod
    def play(self):
        pass

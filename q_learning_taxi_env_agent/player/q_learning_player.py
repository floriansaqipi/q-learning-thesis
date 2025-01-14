from abc import ABC
from typing import override

import numpy as np

from .experience_handler import ExperienceHandler
from ..environment import Environment

from ..constants import PROGRESS_MEMORY_DIRECTORY
from ..agent import QLearningAgent
from .player import Player


class QLearningPlayer(Player, ExperienceHandler, ABC):
    def __init__(self, env: Environment, agent : QLearningAgent, n_episodes: int = None):
        super().__init__(env, n_episodes)
        self.agent = agent

    @override
    def save_progress(self, file_name: str):
        file_full_path = PROGRESS_MEMORY_DIRECTORY + file_name
        q_values_str_keys = {str(k): v for k, v in self.agent.q_values.items()}
        np.savez(file_full_path, **q_values_str_keys)

    @override
    def load_progress(self, file_name: str):
        file_full_path = PROGRESS_MEMORY_DIRECTORY + file_name
        loaded_q_values = np.load(file_full_path)
        self.agent.q_values = {int(key): loaded_q_values[key] for key in loaded_q_values.keys()}


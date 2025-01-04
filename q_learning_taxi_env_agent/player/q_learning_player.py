from abc import ABC
from typing import override

import numpy as np

from .experience_handler import ExperienceHandler
from ..environment import Environment

from ..constants import PROGRESS_MEMORY_FILE_NAME
from ..agent import QLearningAgent
from .player import Player
from ..utils import FileUtils


class QLearningPlayer(Player, ExperienceHandler, ABC):
    def __init__(self, env: Environment, n_episodes: int, agent : QLearningAgent):
        super().__init__(env, n_episodes)
        self.agent = agent

    @override
    def save_progress(self):
        file_path = FileUtils.get_file_path(PROGRESS_MEMORY_FILE_NAME)
        q_values_str_keys = {str(k): v for k, v in self.agent.q_values.items()}
        np.savez(file_path, **q_values_str_keys)

    @override
    def load_progress(self):
        file_path = FileUtils.get_file_path(PROGRESS_MEMORY_FILE_NAME)
        loaded_q_values = np.load(file_path)
        self.agent.q_values = {int(key): loaded_q_values[key] for key in loaded_q_values.keys()}


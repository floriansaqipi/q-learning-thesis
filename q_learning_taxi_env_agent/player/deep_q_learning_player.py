from typing import override

from abc import ABC

import torch

from  .player import Player
from .experience_handler import ExperienceHandler
from ..constants import PROGRESS_MEMORY_DIRECTORY
from ..agent import DeepQLearningAgent
from ..environment import Environment

class DeepQLearningPlayer(Player, ExperienceHandler, ABC):
    def __init__(self, env: Environment, agent : DeepQLearningAgent, n_episodes: int = None):
        super().__init__(env, n_episodes)
        self.agent = agent

    @override
    def save_progress(self, file_name: str):
        file_full_path = PROGRESS_MEMORY_DIRECTORY + file_name
        torch.save(self.agent.q_network.state_dict(), file_full_path)

    @override
    def load_progress(self, file_name: str):
        file_full_path = PROGRESS_MEMORY_DIRECTORY + file_name
        self.agent.q_network.load_state_dict(torch.load(file_full_path, weights_only=True))
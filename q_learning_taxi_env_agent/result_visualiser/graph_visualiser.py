import numpy as np
import matplotlib.pyplot as plt

from .visualiser import Visualiser
from ..environment import StatisticsRecordingEnvironment
from ..agent.agent import Agent

class GraphVisualiser(Visualiser):

    def __init__(
            self,
            env: StatisticsRecordingEnvironment,
            agent: Agent
    ):
        super().__init__()
        self.env = env
        self.agent = agent

    def get_rewards(self):
        return self.env.get_return_queue()

    def get_lengths(self):
        return self.env.get_length_queue()

    def get_training_error(self):
        return self.agent.get_training_error()

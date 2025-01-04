import numpy as np
import matplotlib.pyplot as plt

from .visualiser import Visualiser
from ..environment import StatisticsRecordingEnvironment
from ..agent.agent import Agent

class GraphVisualiser(Visualiser):

    def __init__(self, env: StatisticsRecordingEnvironment, agent: Agent):
        super().__init__(env, agent)

    def visualise(self):
        window = 100
        fig, axs = plt.subplots(1, 3, figsize=(20, 8))

        axs[0].plot(np.convolve(self.env.get_return_queue(), np.ones(window), mode='valid'))
        axs[0].set_title("Episode Rewards")
        axs[0].set_xlabel("Episode")
        axs[0].set_ylabel("Reward")

        axs[1].plot(np.convolve(self.env.get_length_queue(), np.ones(window), mode='valid'))
        axs[1].set_title("Episode Lengths")
        axs[1].set_xlabel("Episode")
        axs[1].set_ylabel("Length")

        axs[2].plot(np.convolve(self.agent.get_training_error(), np.ones(window), mode='valid'))
        axs[2].set_title("Training Error")
        axs[2].set_xlabel("Episode")
        axs[2].set_ylabel("Temporal Difference")

        plt.tight_layout()
        plt.show()

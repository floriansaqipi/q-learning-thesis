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

        self.plot_rewards(axs[0], window)
        self.plot_lengths(axs[1], window)
        self.plot_training_error(axs[2], window)

        plt.tight_layout()
        plt.show()

    def plot_rewards(self, ax, window):

        rewards = self.env.get_return_queue()
        if not rewards:
            ax.set_visible(False)
            return
        rewards_convolved = np.convolve(rewards, np.ones(window), mode='valid')
        ax.plot(rewards_convolved)
        ax.set_title("Episode Rewards")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")

    def plot_lengths(self, ax, window):

        lengths = self.env.get_length_queue()
        if not lengths:
            ax.set_visible(False)
            return
        lengths_convolved = np.convolve(lengths, np.ones(window), mode='valid')
        ax.plot(lengths_convolved)
        ax.set_title("Episode Lengths")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Length")

    def plot_training_error(self, ax, window):

        training_error = self.agent.get_training_error()
        if not training_error:
            ax.set_visible(False)
            return
        training_error_convolved = np.convolve(training_error, np.ones(window), mode='valid')
        ax.plot(training_error_convolved)
        ax.set_title("Training Error")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Temporal Difference")

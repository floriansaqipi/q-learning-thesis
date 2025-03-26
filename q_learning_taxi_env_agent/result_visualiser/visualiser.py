from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np

class Visualiser(ABC):

    def __init__(self):
        self.fig, self.axs = plt.subplots(1, 3, figsize=(20, 8))

    def visualise(self):
        self.plot_rewards()
        self.plot_lengths()
        self.plot_training_error()

        plt.tight_layout()
        plt.show(block=True)

    def plot_rewards(self):
        ax = self.axs[0]
        rewards = self.get_rewards()
        rewards_window_size = int(0.1 * len(rewards))
        if not rewards:
            ax.set_visible(False)
            return
        rewards_convolved = np.convolve(rewards, np.ones(rewards_window_size)/rewards_window_size, mode='valid')
        ax.plot(rewards_convolved)
        ax.set_title("Episode Rewards")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")

    def plot_lengths(self):
        ax = self.axs[1]
        lengths = self.get_lengths()
        lengths_window_size = int(0.1 * len(lengths))
        if not lengths:
            ax.set_visible(False)
            return
        lengths_convolved = np.convolve(lengths, np.ones(lengths_window_size)/lengths_window_size, mode='valid')
        ax.plot(lengths_convolved)
        ax.set_title("Episode Lengths")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Length")

    def plot_training_error(self):
        ax = self.axs[2]
        training_error = self.get_training_error()
        training_error_window_size = int(0.1 * len(training_error))
        if not training_error:
            ax.set_visible(False)
            return
        training_error_convolved = np.convolve(training_error, np.ones(training_error_window_size)/training_error_window_size, mode='valid')
        ax.plot(training_error_convolved)
        ax.set_title("Training Error")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Loss")

    @abstractmethod
    def get_rewards(self):
        pass

    @abstractmethod
    def get_lengths(self):
        pass

    @abstractmethod
    def get_training_error(self):
        pass
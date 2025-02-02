import numpy as np
from typing import override

from .agent import Agent
from ..environment import Environment
from ..constans import Constants

class QLearningAgent(Agent):

    def __init__(self,
            env: Environment,
            learning_rate: float,
            start_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            discount_factor: float):
        super().__init__(env)
        self.q_values = {state: np.zeros(self.env.get_action_space().n) for state in
                         range(self.env.get_observation_space().n)}

        self.learning_rate = learning_rate
        self.epsilon = start_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.discount_factor = discount_factor

    def update(self, obs, action, reward, terminated, next_obs):
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        q_value = reward + (self.discount_factor * future_q_value)
        temporal_difference = q_value - self.q_values[obs][action]
        self.q_values[obs][action] += self.learning_rate * temporal_difference

        self.training_error.append(temporal_difference)

    def get_action(self, obs):
        if np.random.random() < self.epsilon:
            return self.env.random_step()
        else:
            return int(np.argmax(self.q_values[obs]))

    def get_best_action(self, obs):
        return int(np.argmax(self.q_values[obs]))

    def save_progress(self, file_name: str):
        file_full_path = Constants.PROGRESS_MEMORY_DIRECTORY + file_name
        q_values_str_keys = {str(k): v for k, v in self.q_values.items()}
        np.savez(file_full_path, **q_values_str_keys)

    def load_progress(self, file_name: str):
        file_full_path = Constants.PROGRESS_MEMORY_DIRECTORY + file_name
        loaded_q_values = np.load(file_full_path)
        self.q_values = {int(key): loaded_q_values[key] for key in loaded_q_values.keys()}

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)



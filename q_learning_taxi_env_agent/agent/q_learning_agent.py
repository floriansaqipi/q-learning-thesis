import numpy as np
from typing_extensions import override

from .agent import Agent
from .policy import EpsilonGreedyPolicy
from ..environment import Environment

class QLearningAgent(Agent, EpsilonGreedyPolicy):

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


    @override
    def update(self, obs, action, reward, terminated, next_obs):
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        q_value = reward + (self.discount_factor * future_q_value)
        temporal_difference = q_value - self.q_values[obs][action]
        self.q_values[obs][action] += self.learning_rate * temporal_difference

        self.training_error.append(temporal_difference)

    @override
    def get_action(self, obs):
        if np.random.random() < self.epsilon:
            return self.env.random_step()
        else:
            return int(np.argmax(self.q_values[obs]))

    @override
    def get_best_action(self, obs):
        return int(np.argmax(self.q_values[obs]))

    @override
    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

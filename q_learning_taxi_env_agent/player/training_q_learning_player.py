from tqdm import tqdm

from .q_learning_player import QLearningPlayer
from ..agent import QLearningAgent
from ..environment import Environment


class TrainingPlayer(QLearningPlayer):
    def __init__(self,env: Environment, n_episodes, agent: QLearningAgent):
        super().__init__(env, n_episodes, agent)

    def play(self):
        for episode in tqdm(range(self.n_episodes)):
            obs, info = self.env.reset(self.env.seed)
            done = False

            while not done:
                action = self.agent.get_action(obs)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                self.agent.update(obs, action, reward, terminated, next_obs)
                obs = next_obs
                done = terminated or truncated

            self.agent.decay_epsilon()

        self.env.close()


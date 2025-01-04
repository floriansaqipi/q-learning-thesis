
from .q_learning_player import QLearningPlayer
from .. import StatisticsRecordingEnvironment
from ..agent import QLearningAgent
from ..environment import Environment



class InferencePlayer(QLearningPlayer):
    def __init__(self,env: Environment, n_episodes, agent: QLearningAgent):
        super().__init__(env, n_episodes, agent)

    def play(self):
        for episode in range(self.n_episodes):
            obs, info = self.env.reset(self.env.seed)
            done = False

            while not done:
                action = self.agent.get_best_action(obs)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                obs = next_obs
                done = terminated or truncated
        self.env.close()

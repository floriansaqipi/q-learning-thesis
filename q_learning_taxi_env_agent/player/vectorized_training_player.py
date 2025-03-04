from .training_player import TrainingPlayer

class VectorizedTrainingPlayer(TrainingPlayer):

    def play(self):
        episode_count = 0
        obs, info = self.env.reset(self.env.seed)

        while self.n_episodes is None or episode_count < self.n_episodes:
            done  = False

            while not done:
                action = self.agent.get_action(obs)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                self.agent.update(obs, action, reward, terminated, next_obs)
                obs = next_obs
                done = terminated[0] or truncated[0]

            episode_count += 1
            self.progress_bar.update(1)

        self.env.close()
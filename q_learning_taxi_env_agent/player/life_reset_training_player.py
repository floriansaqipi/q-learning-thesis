from .training_player import TrainingPlayer



class LifeResetTrainingPlayer(TrainingPlayer):

    def play(self):
        episode_count = 0


        while self.n_episodes is None or episode_count < self.n_episodes:
            obs, info = self.env.reset(self.env.seed)
            prev_lives = info.get("lives", 0)
            obs, _, _, _, info = self.env.step(1)
            done = False

            while not done:
                action = self.agent.get_action(obs)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                current_lives = info.get('lives', 0)

                if current_lives < prev_lives:
                    next_obs, _, _, _, info = self.env.step(1)
                    prev_lives = current_lives
                else:
                    self.agent.update(obs, action, reward, terminated, next_obs)

                obs = next_obs
                done = terminated or truncated

            self.agent.end_of_episode_hook()

            episode_count += 1
            if self.printing_enabled:
                self.progress_bar.update(1)

            if self.save_frequency is not None and episode_count % self.save_frequency == 0:
                self.save_progress(self.save_frequency, episode_count)

        self.env.close()


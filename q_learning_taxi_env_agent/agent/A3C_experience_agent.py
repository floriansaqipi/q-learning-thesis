from .A3C_agent import A3CAgent
from .utils import JsonProgressHandler, ExperienceHandler
from ..constans import A3CConstants
from ..environment import EnvironmentWrapper, StatisticsRecordingEnvironment


class A3CExperienceAgent(A3CAgent):


    def save_progress(self, save_frequency: int = 0, episode_number: int = 0, return_queue=None, length_queue=None):
        self.a3c_global_network.save_progress(save_frequency, episode_number, return_queue, length_queue)
        JsonProgressHandler.save_episode_number(save_frequency, episode_number,
                                                A3CConstants.PROGRESS_EPISODE_NUMBER_FILE_NAME)
        temp_env = self.env
        while isinstance(temp_env, EnvironmentWrapper):
            if isinstance(temp_env, StatisticsRecordingEnvironment):
                JsonProgressHandler.save_statistics(
                    temp_env.get_return_queue(),
                    temp_env.get_length_queue(),
                    self.training_error,
                    A3CConstants.PROGRESS_STATISTICS_FILE_NAME
                )
                break
            temp_env = temp_env.unwrap()

    def load_progress(self):
        pass
import gymnasium as gym

from .environment_wrapper import EnvironmentWrapper
from .environment import Environment
from ..constans import Constants


class VideoRecordingEnvironment(EnvironmentWrapper):
    def __init__(self, env: Environment, record_frequency: int, video_directory_name: str,
                 name_prefix: str = Constants.PLAYER_NAME_PREFIX):
        super().__init__(env)
        self.record_frequency = record_frequency
        self.video_directory_name = video_directory_name
        self.name_prefix = name_prefix
        self.inner_env = gym.wrappers.RecordVideo(
            self.env.inner_env,
            video_folder=Constants.PLAYER_VIDEOS_DIRECTORY + self.video_directory_name, name_prefix=self.name_prefix,
            episode_trigger=EpisodeTrigger(record_frequency)
        )

class EpisodeTrigger:
    def __init__(self, record_frequency: int):
        self.record_frequency = record_frequency

    def __call__(self, episode_number: int) -> bool:
        return episode_number % self.record_frequency == 0

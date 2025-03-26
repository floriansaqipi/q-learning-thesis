from .constants import Constants


class A3CConstants(Constants):
    PROGRESS_MEMORY_FILE_NAME = "A3C_network_checkpoint.pth"
    PROGRESS_STATISTICS_FILE_NAME = "A3C_statistics_checkpoint.json"
    PROGRESS_EPISODE_NUMBER_FILE_NAME = "A3C_episode_number.json"
    TRAINING_PLAYER_VIDEOS_DIRECTORY_NAME = "training_A3C_player_videos"
    INFERENCE_PLAYER_VIDEOS_DIRECTORY_NAME = "inference_A3C_player_videos"
    PLAYER_NAME_PREFIX = "A3C_learning_player"

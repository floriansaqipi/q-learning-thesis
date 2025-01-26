from .constants import Constants

class QLearningConstants(Constants):
    PROGRESS_MEMORY_FILE_NAME = "q_values.npz"
    TRAINING_PLAYER_VIDEOS_DIRECTORY_NAME = "training_q_learning_player_videos"
    INFERENCE_PLAYER_VIDEOS_DIRECTORY_NAME = "inference_q_learning_player_videos"
    PLAYER_NAME_PREFIX = "q_learning_player"
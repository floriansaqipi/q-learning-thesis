import torch.multiprocessing as mp

from q_learning_taxi_env_agent import Environment, A3CConstants, RenderMode, StatisticsRecordingEnvironment, \
    VideoRecordingEnvironment, A3CAgent, LifeResetTrainingPlayer, InferencePlayer
from q_learning_taxi_env_agent.agent.A3C_experience_agent import A3CExperienceAgent
from q_learning_taxi_env_agent.agent.utils import A3CGlobalNetwork
from q_learning_taxi_env_agent.hyper_parameters import DECAY_FACTOR, EPSILON, VALUE_LOSS_COEFFICIENT
from q_learning_taxi_env_agent.hyper_parameters.A3C_hyper_parameters import N_TRAINING_EPISODES, N_WORKERS, \
    LEARNING_RATE, DISCOUNT_FACTOR, ENTROPY_REGULARIZATION_COEFFICIENT, N_STEPS, MAX_NORM
from q_learning_taxi_env_agent.result_visualiser.offline_graph_visualiser import OfflineGraphVisualiser

RECORD_FREQUENCY = N_TRAINING_EPISODES / 10

if __name__ == "__main__":

    env = Environment(A3CConstants.ENVIRONMENT_ALE_BREAKOUT_NO_FRAMESKIP_V4, RenderMode.RGB_ARRAY)

    a3c_global_network = A3CGlobalNetwork(env, LEARNING_RATE, DECAY_FACTOR, EPSILON)
    a3c_global_network.load_progress()

    statistics_recording_env = StatisticsRecordingEnvironment(env, N_TRAINING_EPISODES)
    video_recording_env = VideoRecordingEnvironment(statistics_recording_env, RECORD_FREQUENCY,
                                                    A3CConstants.TRAINING_PLAYER_VIDEOS_DIRECTORY_NAME)

    agent = A3CExperienceAgent(video_recording_env, a3c_global_network, DISCOUNT_FACTOR,
                               ENTROPY_REGULARIZATION_COEFFICIENT, VALUE_LOSS_COEFFICIENT, N_STEPS, MAX_NORM, True)

    player = InferencePlayer(agent, video_recording_env, N_TRAINING_EPISODES)
    player.play()


    visualiser = OfflineGraphVisualiser(A3CConstants.PROGRESS_STATISTICS_FILE_NAME)
    visualiser.visualise()

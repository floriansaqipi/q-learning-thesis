from q_learning_taxi_env_agent import Environment, A2CConstants, RenderMode, StatisticsRecordingEnvironment, \
    VideoRecordingEnvironment, A2CAgent, VectorizedTrainingPlayer, VectorizedEnvironment

from q_learning_taxi_env_agent.hyper_parameters.A2C_hyper_parameters import N_TRAINING_EPISODES, N_ENVIRONMENTS, \
    LEARNING_RATE, DISCOUNT_FACTOR, ENTROPY_REGULARIZATION_COEFFICIENT
from q_learning_taxi_env_agent.result_visualiser.offline_graph_visualiser import OfflineGraphVisualiser

RECORD_FREQUENCY = N_TRAINING_EPISODES / 10
SAVING_FREQUENCY = N_TRAINING_EPISODES / 100

if __name__ == "__main__":

    env = Environment(A2CConstants.ENVIRONMENT_ALE_BREAKOUT_NO_FRAMESKIP_V4, RenderMode.RGB_ARRAY)
    statistics_recording_env = StatisticsRecordingEnvironment(env, N_TRAINING_EPISODES)
    video_recording_env = VideoRecordingEnvironment(statistics_recording_env, RECORD_FREQUENCY,
                                                    A2CConstants.TRAINING_PLAYER_VIDEOS_DIRECTORY_NAME)

    vectorized_env = VectorizedEnvironment(video_recording_env, N_ENVIRONMENTS)

    agent = A2CAgent(vectorized_env, LEARNING_RATE, DISCOUNT_FACTOR, ENTROPY_REGULARIZATION_COEFFICIENT, statistics_recording_env)

    player = VectorizedTrainingPlayer(agent, vectorized_env, N_TRAINING_EPISODES, SAVING_FREQUENCY)
    player.load_progress()
    player.play()

    visualiser = OfflineGraphVisualiser(A2CConstants.PROGRESS_STATISTICS_FILE_NAME)
    visualiser.visualise()

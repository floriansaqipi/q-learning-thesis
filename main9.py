from q_learning_taxi_env_agent import Environment, A3CConstants, RenderMode, StatisticsRecordingEnvironment, \
    VideoRecordingEnvironment, A3CAgent, VectorizedTrainingPlayer, GraphVisualiser, VectorizedEnvironment

from q_learning_taxi_env_agent.hyper_parameters.A3C_hyper_parameters import N_TRAINING_EPISODES, N_ENVIRONMENTS, \
    LEARNING_RATE, DISCOUNT_FACTOR, ENTROPY_REGULARIZATION_COEFFICIENT

RECORD_FREQUENCY = N_TRAINING_EPISODES / 10

if __name__ == "__main__":

    env = Environment(A3CConstants.ENVIRONMENT_ALE_BREAKOUT_V5, RenderMode.RGB_ARRAY)
    statistics_recording_env = StatisticsRecordingEnvironment(env, N_TRAINING_EPISODES)
    video_recording_env = VideoRecordingEnvironment(statistics_recording_env, RECORD_FREQUENCY,
                                                    A3CConstants.TRAINING_PLAYER_VIDEOS_DIRECTORY_NAME)

    vectorized_env = VectorizedEnvironment(video_recording_env, 10)

    agent = A3CAgent(vectorized_env, LEARNING_RATE, DISCOUNT_FACTOR, ENTROPY_REGULARIZATION_COEFFICIENT)

    player = VectorizedTrainingPlayer(agent, vectorized_env, N_TRAINING_EPISODES)
    player.play()
    player.save_progress(A3CConstants.PROGRESS_MEMORY_FILE_NAME)

    visualiser = GraphVisualiser(statistics_recording_env, agent)
    visualiser.visualise()

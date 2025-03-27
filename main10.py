from q_learning_taxi_env_agent import Environment, A2CConstants, RenderMode, StatisticsRecordingEnvironment, \
    VideoRecordingEnvironment, A2CAgent, GraphVisualiser, InferencePlayer

from q_learning_taxi_env_agent.hyper_parameters.A2C_hyper_parameters import N_PLAYING_EPISODES, \
    LEARNING_RATE, DISCOUNT_FACTOR, ENTROPY_REGULARIZATION_COEFFICIENT

RECORD_FREQUENCY = N_PLAYING_EPISODES / 10

if __name__ == "__main__":

    env = Environment(A2CConstants.ENVIRONMENT_ALE_BREAKOUT_NO_FRAMESKIP_V4, RenderMode.RGB_ARRAY)
    statistics_recording_env = StatisticsRecordingEnvironment(env, N_PLAYING_EPISODES)
    video_recording_env = VideoRecordingEnvironment(statistics_recording_env, RECORD_FREQUENCY,
                                                    A2CConstants.INFERENCE_PLAYER_VIDEOS_DIRECTORY_NAME)



    agent = A2CAgent(video_recording_env, LEARNING_RATE, DISCOUNT_FACTOR, ENTROPY_REGULARIZATION_COEFFICIENT)

    player = InferencePlayer(agent, video_recording_env, N_PLAYING_EPISODES)
    player.load_progress()
    player.play()

    visualiser = GraphVisualiser(statistics_recording_env, agent)
    visualiser.visualise()



from q_learning_taxi_env_agent import Environment, VideoRecordingEnvironment, QLearningAgent, \
    GraphVisualiser, RenderMode, StatisticsRecordingEnvironment, InferencePlayer, QLearningConstants

from q_learning_taxi_env_agent.hyper_parameters.q_learning_hyper_parameters import N_TRAINING_EPISODES, N_PLAYING_EPISODES, SEED, LEARNING_RATE, \
    START_EPSILON, EPSILON_DECAY, \
    FINAL_EPSILON, DISCOUNT_FACTOR

RECORD_FREQUENCY = N_PLAYING_EPISODES / 10

env = Environment(QLearningConstants.ENVIRONMENT_ID_TAXI_V3, RenderMode.RGB_ARRAY, SEED)
statistics_recording_env = StatisticsRecordingEnvironment(env, N_TRAINING_EPISODES)
video_recording_env = VideoRecordingEnvironment(statistics_recording_env, RECORD_FREQUENCY, QLearningConstants.INFERENCE_PLAYER_VIDEOS_DIRECTORY_NAME)

agent = QLearningAgent(video_recording_env, LEARNING_RATE, START_EPSILON, EPSILON_DECAY, FINAL_EPSILON, DISCOUNT_FACTOR)

player = InferencePlayer(agent, video_recording_env, N_PLAYING_EPISODES)
player.load_progress()
player.play()



visualiser = GraphVisualiser(statistics_recording_env, agent)
visualiser.visualise()

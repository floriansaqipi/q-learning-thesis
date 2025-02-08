

from q_learning_taxi_env_agent import Environment, VideoRecordingEnvironment, QLearningAgent, TrainingPlayer, \
    GraphVisualiser, RenderMode, StatisticsRecordingEnvironment, QLearningConstants

from q_learning_taxi_env_agent.hyper_parameters.q_learning_hyper_parameters import N_TRAINING_EPISODES, SEED, LEARNING_RATE, \
    START_EPSILON, EPSILON_DECAY, \
    FINAL_EPSILON, DISCOUNT_FACTOR

RECORD_FREQUENCY = N_TRAINING_EPISODES / 10

env = Environment(QLearningConstants.ENVIRONMENT_ID_TAXI_V3, RenderMode.RGB_ARRAY, SEED)
statistics_recording_env = StatisticsRecordingEnvironment(env, N_TRAINING_EPISODES)
video_recording_env = VideoRecordingEnvironment(statistics_recording_env, RECORD_FREQUENCY, QLearningConstants.TRAINING_PLAYER_VIDEOS_DIRECTORY_NAME)

agent = QLearningAgent(video_recording_env, LEARNING_RATE, START_EPSILON, EPSILON_DECAY, FINAL_EPSILON, DISCOUNT_FACTOR)

player = TrainingPlayer( agent, video_recording_env, N_TRAINING_EPISODES)
player.play()
player.save_progress(QLearningConstants.PROGRESS_MEMORY_FILE_NAME)


visualiser = GraphVisualiser(statistics_recording_env, agent)
visualiser.visualise()

from q_learning_taxi_env_agent import Environment, DeepQLearningAgent, InferencePlayer, \
     RenderMode, DeepQLearningConstants


from q_learning_taxi_env_agent.agent import ReplayMemory


from q_learning_taxi_env_agent.hyper_parameters.deep_q_learning_hyper_parameters import LEARNING_RATE, \
    START_EPSILON, EPSILON_DECAY, \
    FINAL_EPSILON, DISCOUNT_FACTOR, INTERPOLATION_PARAMETER, REPLAY_MEMORY_SIZE, REPLAY_MEMORY_BATCH_SIZE


env = Environment(DeepQLearningConstants.ENVIRONMENT_LUNAR_LANDER_V3, RenderMode.HUMAN)

replay_memory = ReplayMemory(REPLAY_MEMORY_SIZE, REPLAY_MEMORY_BATCH_SIZE)

agent = DeepQLearningAgent(env, LEARNING_RATE, DISCOUNT_FACTOR, INTERPOLATION_PARAMETER, START_EPSILON, EPSILON_DECAY, FINAL_EPSILON, replay_memory)

player = InferencePlayer(agent, env)
player.load_progress(DeepQLearningConstants.PROGRESS_MEMORY_FILE_NAME)
player.play()

import gymnasium as gym
import ale_py

gym.register_envs(ale_py)
env = gym.make("BreakoutNoFrameskip-v4", render_mode="human")  # Replace with your specific environment ID
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=1000)


observation, info = env.reset()
cnt = 0
for _ in range(1000):
    episode_over = False
    flag = False
    action = 0

    while not episode_over:

        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        episode_over = terminated or truncated
        cnt = cnt + 1


    env.close()


print(env.spec.max_episode_steps)

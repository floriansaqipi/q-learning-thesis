import gymnasium as gym
import ale_py

gym.register_envs(ale_py)
env = gym.make("BreakoutDeterministic-v4", render_mode="human")  # Replace with your specific environment ID


observation, info = env.reset()

for _ in range(1000):
    episode_over = False
    flag = False
    action = 0
    while not episode_over:

        action = env.action_space.sample()
        while action == 1 and flag:
            action = env.action_space.sample()  # agent policy that uses the observation and info
        flag = True
        observation, reward, terminated, truncated, info = env.step(action)

        episode_over = terminated or truncated
        if episode_over:
            print("smth")


    env.close()


print(env.spec.max_episode_steps)
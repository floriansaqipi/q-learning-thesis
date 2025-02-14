import gymnasium as gym
import ale_py


gym.register_envs(ale_py)

env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
print (env.spec.max_episode_steps)
env.reset()
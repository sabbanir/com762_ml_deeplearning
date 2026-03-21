

import gymnasium as gym
import Box2D
env = gym.make("LunarLander-v3", render_mode="human")
# env = gym.make('MountainCar-v0')
observation, info = env.reset(seed=42)
for _ in range(5000):
   action = env.action_space.sample()  # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()

env.close()
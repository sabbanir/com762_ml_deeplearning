
import gymnasium as gym
import pygame
env = gym.make('MountainCar-v0', render_mode="human")
# env = gym.make("CartPole-v1", render_mode="human")

for i_episode in range(20):
    observation, info = env.reset(seed=42)
    for t in range(100):
        action = 2 if observation[1] > 0 else 0
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()


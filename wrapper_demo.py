import gym
from wrapper import make_env
import numpy as np


obs_cost = 0.1
obs_flag = 1
vanilla = 0

env = make_env("CartPole-v1", obs_cost, obs_flag, vanilla)

observation = env.reset()
for _ in range(100):
  env.render()
  action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action)
  print(done)
  if done:
    observation = env.reset()
    print("..........")


env.close()

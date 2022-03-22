**Active-measure MDP with explicit cost wrapper for the Open AI Gym. **

**To cite the framework and find more details:**

@inproceedings{bellinger2022active-measure,
  title={Balancing Information with Observation Costs in Deep Reinforcement Learning},
  author={Bellinger, Colin and Drozdyuk, Andriy and Crowley, Mark and Tamblyn, Isaac},
  booktitle={Canadian Conference on Artificial Intelligence},
  year={2022}
}

**Demonstration of the wrapper class applied to the cartpole gym environment:**

import gym
from wrapper import make_env
import numpy as np


obs_cost = 0.1
obs_flag = 1
vanilla = 0

env = make_env("CartPole-v1", obs_cost, obs_flag, vanilla)

observation = env.reset()
for _ in range(1000):
  
  env.render()
  action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action)
  
  
  if done:
    
    observation = env.reset()

env.close()

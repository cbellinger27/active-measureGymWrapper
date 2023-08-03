# Active-measure MDP with explicit cost wrapper for the Open AI Gym.

This repository includes the code associated with the paper 'Balancing Information with Observation Costs in Deep Reinforcement Learning' presented at the Candadian  Conference of AI 2022. See the manuscript [here](https://caiac.pubpub.org/pub/0jmy7gpd/release/1).

# Citation

  @inproceedings{bellinger2022active-measure,
  title={Balancing Information with Observation Costs in Deep Reinforcement Learning},
  author={Bellinger, Colin and Drozdyuk, Andriy and Crowley, Mark and Tamblyn, Isaac},
  booktitle={Canadian Conference on Artificial Intelligence},
  year={2022}
  }

# Overview

The provided gym wrapper environment converts standard OpenAI Gym environments into environments where at each time step the agent select a control action (such as move left, increase torque, etc.) and decides whether or not to measure the next state of the environment. To achieve this behaviour, the new action space is the cross product of the orininal discrete action space and the choice to measure or not measure (1,0). Whenever a measurement is made, the intrinsic measurement cost is subtracted from the extrinsic control reward and returned to the agent. 

# Demonstration of the wrapper class applied to the cartpole gym environment:

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

# Demonstration of the wrapper class with Stable Baselines:

  import gym
  from wrapper import make_env
  from stable_baselines3 import DQN

  obs_cost = 0.1
  obs_flag = 1
  vanilla = 0
  
  env = make_env("CartPole-v1", obs_cost, obs_flag, vanilla)
  model = DQN("MlpPolicy", env, verbose=1)
  model.learn(total_timesteps=int(2e5), progress_bar=True)
  model.save("dqn_lunar")

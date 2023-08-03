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
    import pandas as pd
    import matplotlib.pyplot as plt

    obs_cost = 0.1
    obs_flag = 1
    vanilla = 0

    env = make_env("CartPole-v1", obs_cost, obs_flag, vanilla)
    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=50000)
        
    obs = env.reset()
    num_measure = 0
    num_noMeasure = 0
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        if env.is_measure_action(action):
          num_measure += 1
        else:
          num_noMeasure += 1
        _ = env.render("human")
        if done:
          obs = env.reset()

    df = pd.DataFrame({'Total steps':[1000], 'Total Measurements': [num_measure], 'Total No Measurement': [num_noMeasure]})
    df.plot.bar(title='CartPole Policy Rollout')

import gym
import numpy as np
from gym.spaces import Box

  
def make_env(env_name, obs_cost, obs_flag, vanilla):
    """
    Make an environment, potentially wrapping it in MeasureWrapper.
    
    Args:
        env_name: Environment name
        obs_cost: Observation cost to use for the wrapper
        obs_flag: Wether to include a flag in the output to indicate if last action was observe
        vanilla: If True, uses the original environment without a wrapper. Ignores obs_cost and obs_flag arguments.

    Returns:
        A gym environment.
    """
    env = gym.make(env_name)
    if vanilla:
        return VanillaWrapper(env)
    else:
        env = MeasureWrapper(env, obs_cost=obs_cost, unknown_state='LAST_MEASURED', obs_flag=obs_flag)
        return env

class VanillaWrapper(gym.Wrapper):
    def is_measure_action(self, _action):
        return False
    
    def step(self, action):
        state, reward, done, info = self.env.step(action)

        info['cost'] = 0.0
        info['reward'] = reward

        return state, reward, done, info

class MeasureWrapper(gym.Wrapper):
    """Augments environments with an extra observe action and optionally an observe flag.
    Stores the original reward in the info['reward'] attribute, and the cost in the info['cost'] attribute.

    "measure" either 0 or 1. If it is 1 then action (below) is carried
    out and an accurate state representation is returned with some penalty
    to the reward. If 0, then no penalty is incurred to reward but the state
    returned cannot be trusted to be accurate.
    
    For discrete environments, measure action is encoded as:
    1.    0,   (n-1) - don't measure, just carry out the action
    2. (n+1), (2n-1) - measure, carry out the (action - n)

    So, for example, in CartPole there are two actions, so n=2:
    1. 0-1 - don't measure
    2. 2-3 - measure 

    In AcroBot, n=3, so:
    1. 0-2 - don't measure
    2. 3-5 - measure
    
    For continuous environments, a new bit is prepended to the action tensor, e.g.:

        action = [1,2,3]
        measure_action = [0,1,2,3]
    
    If obs_flag is True then state is augmented (prepended) 
    with observation flag, which is 1 if measure action was taken and 0 otherwise.
    
    Unknown state is either RANDOM, LAST_MEASURED or NOISY.
    RANDOM is a random sample from observation space.
    LAST_MEASURED returns the state that was last measured.
    NOISY_X samples noise with X stdev from a normal distribution with a mean of 0.
    """
    def __init__(self, env, obs_cost, unknown_state, obs_flag=True):
        super().__init__(env)
        self.obs_flag = obs_flag
        self.continuous_action_space = True if env.action_space.shape else False
                
        if self.continuous_action_space:
            la = np.concatenate(([0], env.unwrapped.action_space.low))
            ha = np.concatenate(([1], env.unwrapped.action_space.high))        
            self.action_space = Box(low=la, high=ha, 
                dtype=env.unwrapped.action_space.dtype)
        else:
            self.action_space = gym.spaces.Discrete(2*env.action_space.n)

        self.state_low = env.observation_space.low
        self.state_high = env.observation_space.high

        self.unknown_state = unknown_state
        self.last_state = None
        self.obs_cost = obs_cost    

        l = env.unwrapped.observation_space.low
        h = env.unwrapped.observation_space.high  
        if self.obs_flag:
            l = np.concatenate(([0], l))
            h = np.concatenate(([1], h))        
        self.observation_space = Box(low=l, high=h)

    def is_measure_action(self, action):
        if self.continuous_action_space:
            return action[0] >= 0.5
        else:
            return action >= self.env.action_space.n
        
    def step(self, action):        
        if self.is_measure_action(action):
            measure = 1
            
            if not self.continuous_action_space:
                action = action - self.env.action_space.n
        else:
            measure = 0
        
        if self.continuous_action_space:
            action = action[1:]

        state, reward, done, info = self.env.step(action)
        env_reward = reward
       
        if measure == 1:
            reward -= self.obs_cost
            self.last_state = state
        elif measure == 0:
            if self.unknown_state == 'RANDOM':
                state = self.unwrapped.observation_space.sample()
            elif self.unknown_state == 'LAST_MEASURED':
                state = self.last_state
            elif self.unknown_state.startswith('NOISY'):
                std = self.parse_std(self.unknown_state)
                noise = np.random.normal(loc=state, scale=std)
                state = np.clip(state + noise, self.state_low, self.state_high)            
            else:
                raise ValueError(f"Unknown state invalid: {self.unknown_state}")

        if self.obs_flag:    
            state = np.concatenate(([measure], state))
        

        info['cost'] = measure * self.obs_cost
        info['reward'] = env_reward
        return state, reward, done, info

    def parse_std(self, unknown_state):
        return float(unknown_state.split('-')[1])


    def reset(self):
        state = self.env.reset()
        self.last_state = state

        if self.obs_flag:
            state = np.concatenate(([1], state))

        return state
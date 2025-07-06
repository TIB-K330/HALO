import gym
import numpy as np

class SimpleEnv(gym.Env):
    """
    For setting action space only 

    """
    def __init__(self) -> None:
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

class DiscreteEnv(gym.Env):
    """
    For setting action space only 

    """
    def __init__(self) -> None:
        self.action_space = gym.spaces.Discrete(81)
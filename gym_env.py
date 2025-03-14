import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import NormalizeReward
from Simulator.gui.sim import Simulator

class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 100}

    def __init__(self, render_mode=None):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        seed = 42
        self.sim_env = Simulator(render_fps=self.metadata["render_fps"], seed=seed)
        # Example when using discrete actions:
        self.action_space = spaces.Box(low=-1, high=1,
                                            shape=(3,), dtype=np.float32)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=-5, high=5,
                                            shape=(26,), dtype=np.float32)
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        # if render_mode is not None:
        #     self.sim_env.init_window()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.sim_env.step(action)
        # if self.render_mode is not None:
        self.render()
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        observation, info = self.sim_env.reset()
        return observation, info

    def render(self):
        self.sim_env.render()

    def close(self):
        self.sim_env.kill_window()
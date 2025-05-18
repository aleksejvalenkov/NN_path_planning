import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import NormalizeReward
from Simulator.gui.sim import Simulator

class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, render_mode=None, **kwargs):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects

        self.PID_MODE = False
        self.sim_env = Simulator(render_fps=self.metadata["render_fps"], 
                                                seed=kwargs['seed'],
                                                robot_init_pos=kwargs['robot_init_pos'],
                                                robot_goal_pos=kwargs['robot_goal_pos'])
        
        observation, info = self.sim_env.reset()
        print('observation shape = ', info['shape'])
        # Example when using discrete actions:
        self.action_space = spaces.Box(low=-1, high=1,
                                            shape=(3,), dtype=np.float32)
        # Example for using image as input (channel-first; channel-last also works):
        # self.observation_space = spaces.Box(low=-10, high=10,
        #                                     shape=info['shape'], dtype=np.float32)
        
        self.observation_space = spaces.Dict({
            'ranges': spaces.Box(low=0, high=1, shape=(20,), dtype=np.float32),
            # 'ranges': spaces.Box(low=0, high=1, shape=(180,), dtype=np.float32),
            'velocity': spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
            'target_point_vector': spaces.Box(low=-10, high=10, shape=(2,), dtype=np.float32),
            'robot_orientation': spaces.Box(low=-4, high=4, shape=(1,), dtype=np.float32),
            'target_orientation': spaces.Box(low=-4, high=4, shape=(1,), dtype=np.float32),
        })
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        # if render_mode is not None:
        #     self.sim_env.init_window()
        

    def step(self, action):
        # print("action =", action)
        observation, reward, terminated, truncated, info = self.sim_env.step(action, pid_mode=self.PID_MODE)
        # print("observation = ", observation)
        # print(f"pid_mode: {self.PID_MODE}")
        # if self.render_mode is not None:
        self.render()
        # print("reward = ", type(reward), reward)
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        observation, info = self.sim_env.reset()
        return observation, info

    def render(self):
        self.sim_env.render()

    def close(self):
        self.sim_env.kill_window()

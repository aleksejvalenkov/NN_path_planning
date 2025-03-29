import os
import sys
import numpy as np
from tqdm import tqdm

import gymnasium as gym
from PPO.env.gym_env import CustomEnv
from PPO.agent.agent import Agent

# Define the initial and goal position of the robot
robot_init_pos = [100, 100, 1.57]
robot_goal_pos = [1500, 300, 1.57]
# robot_init_pos = None
# robot_goal_pos = None

# load and wrap the gymnasium environment.
NUM_ENVS = 1
gym.register(id="my_v1", entry_point=CustomEnv, vector_entry_point=CustomEnv)
env = gym.make(id="my_v1", 
               render_mode=None, 
               seed=42,
               robot_init_pos=robot_init_pos,
               robot_goal_pos=robot_goal_pos,
               )

device = 'cuda:0'

agent = Agent(env.observation_space, env.action_space, device=device)




goal_reached = 0
collision_static = 0
collision_moveable = 0
time_is_out = 0
mean_done_time = 0

observation, info = env.reset()
while True:
    action = agent.gen_action(observation)
    observation, reward, terminated, truncated, info = env.step([0,0,0])
    if terminated:
        print(reward)



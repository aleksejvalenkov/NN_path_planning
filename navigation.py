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
robot_init_pos = None
robot_goal_pos = None

# load and wrap the gymnasium environment.
NUM_ENVS = 1
gym.register(id="my_v1", entry_point=CustomEnv, vector_entry_point=CustomEnv)
env = gym.make(id="my_v1", 
               render_mode=None, 
               seed=None,
               robot_init_pos=robot_init_pos,
               robot_goal_pos=robot_goal_pos,
               )

device = 'cuda:0'

agent = Agent(env.observation_space, env.action_space, device=device)



NUM_EPISODES = 3
goal_reached = 0
collision_static = 0
collision_moveable = 0
time_is_out = 0
mean_done_time = 0

for episode_id in tqdm(range(NUM_EPISODES)):
    # reset the environment and new episode
    observation, info = env.reset()
    while True:
        action = agent.gen_action(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            if info['reason'] == 'Goal reached':
                goal_reached += 1
                mean_done_time += info['done_time']
            elif info['reason'] == 'Collision':
                if info['obstacle_type'] == 'moveable':
                    collision_moveable += 1
                else:
                    collision_static += 1
            elif info['reason'] == 'Time is out':
                time_is_out += 1
            break

# collision_count = collision_static + collision_moveable
# print(f"Goal reached: {goal_reached}/{NUM_EPISODES} = {goal_reached/NUM_EPISODES}")
# print(f"Collision: {collision_count}/{NUM_EPISODES} = {collision_count/NUM_EPISODES}")
# print(f"Collision static: {collision_static}/{collision_count} = {collision_static/collision_count}")
# print(f"Collision moveable: {collision_moveable}/{collision_count} = {collision_moveable/collision_count}")
# print(f"Time is out: {time_is_out}/{NUM_EPISODES} = {time_is_out/NUM_EPISODES}")
# print(f"Mean done time: {mean_done_time/goal_reached}")

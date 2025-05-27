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
                                                robot_goal_pos=kwargs['robot_goal_pos'],
                                                run_dwa=kwargs['run_dwa'])
        
        self.evolution = kwargs['evolution'] if 'evolution' in kwargs else False

        self.observation, info = self.sim_env.reset()
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
        self.episode_id = 0
        self.num_episodes = 3
        self.goal_reached = 0
        self.collision_static = 0
        self.collision_moveable = 0
        self.time_is_out = 0
        self.mean_done_time = 0
        self.reaced_waypoints = 0
        

    def step(self, action):
        # print("action =", action)
        observation, reward, terminated, truncated, info = self.sim_env.step(action, pid_mode=self.PID_MODE)

        if self.evolution:
            if terminated or truncated:
                self.episode_id += 1
                if info['reason'] == 'Goal reached':
                    self.goal_reached += 1
                    self.mean_done_time += info['done_time']
                elif info['reason'] == 'Collision':
                    if info['obstacle_type'] == 'moveable':
                        self.collision_moveable += 1
                    else:
                        self.collision_static += 1
                elif info['reason'] == 'Time is out':
                    self.time_is_out += 1
                
                self.reaced_waypoints += info["target_reached"]/info["max_path_length"]
                self.reset()

        # if self.render_mode is not None:
        self.render()
        # print("reward = ", type(reward), reward)
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        if self.evolution:
            print(f"Episode {self.episode_id} finished")
            if self.episode_id >= self.num_episodes:
                collision_count = self.collision_static + self.collision_moveable
                print(f"Goal reached: {self.goal_reached}/{self.episode_id} = {self.goal_reached/self.episode_id}")
                print(f"Collision: {collision_count}/{self.episode_id} = {collision_count/self.episode_id}")
                if collision_count > 0:
                    print(f"Collision static: {self.collision_static}/{collision_count} = {self.collision_static/collision_count}")
                    print(f"Collision moveable: {self.collision_moveable}/{collision_count} = {self.collision_moveable/collision_count}")
                else:
                    print(f"Collision static: {self.collision_static}/{collision_count} = {0}")
                    print(f"Collision moveable: {self.collision_moveable}/{collision_count} = {0}")
                print(f"Time is out: {self.time_is_out}/{self.num_episodes} = {self.time_is_out/self.num_episodes}")
                if self.goal_reached > 0:
                    print(f"Mean done time: {self.mean_done_time/self.goal_reached} sec")
                else:
                    print(f"Mean done time: {0} sec")
                print(f"Mean reached waypoints: {self.reaced_waypoints}/{self.episode_id} = {self.reaced_waypoints/self.episode_id} min")


        observation, info = self.sim_env.reset()
        return observation, info

    def render(self):
        self.sim_env.render()

    def close(self):
        self.sim_env.kill_window()

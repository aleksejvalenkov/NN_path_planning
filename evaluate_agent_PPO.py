import os
import numpy as np
from tqdm import tqdm

import gymnasium as gym
from gymnasium import spaces

import torch
# import the skrl components to build the RL system
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.trainers.torch import ParallelTrainer
from skrl.utils import set_seed

from gym_env import CustomEnv

from skrl_policy import Policy, Value

# load and wrap the gymnasium environment.
NUM_ENVS = 1
# custom_env = CustomEnv(render_mode="human")
# custom_env = CustomEnv(render_mode=None)

gym.register(id="my_v1", entry_point=CustomEnv, vector_entry_point=CustomEnv)
# env = gym.make_vec(id="my_v1", num_envs=NUM_ENVS, vectorization_mode="async")
env = gym.make(id="my_v1")

# env = wrap_env(env)
# device = env.device
# print(device)
device = 'cuda:0'

# instantiate a memory as rollout buffer (any memory can be used for this)
# memory = RandomMemory(memory_size=1024, num_envs=NUM_ENVS, device=device)
memory = RandomMemory(memory_size=1024, device=device)


# instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
models = {}
models["policy"] = Policy(env.observation_space, env.action_space, device, clip_actions=True)
models["value"] = Value(env.observation_space, env.action_space, device)


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 1024  # memory_size
cfg["learning_epochs"] = 10
cfg["mini_batches"] = 32
cfg["discount_factor"] = 0.9
cfg["lambda"] = 0.95
cfg["learning_rate"] = 1e-2
cfg["learning_rate_scheduler"] = KLAdaptiveRL
cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
cfg["grad_norm_clip"] = 0.5
cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = False
cfg["entropy_loss_scale"] = 0.0
cfg["value_loss_scale"] = 0.5
cfg["kl_threshold"] = 0
cfg["mixed_precision"] = True
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 500
cfg["experiment"]["checkpoint_interval"] = 50000
cfg["experiment"]["directory"] = "runs/torch/metric_env_and_ansiolute_penalty_evaluate"

agent = PPO(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)

agent.load('runs/torch/metric_env_and_ansiolute_penalty/25-03-14_12-15-10-341140_PPO/checkpoints/agent_350000.pt')

NUM_EPISODES = 100
goal_reached = 0
collision_static = 0
collision_moveable = 0
time_is_out = 0
mean_done_time = 0

for episode_id in tqdm(range(NUM_EPISODES)):
    # reset the environment and new episode
    observation, info = env.reset()
    while True:
        observation = torch.tensor(observation, dtype=torch.float32, device=device)
        action = agent.act(observation, timestep=0, timesteps=1)
        action_np = action[0].cpu().detach().numpy()
        observation, reward, terminated, truncated, info = env.step(action_np)
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

collision_count = collision_static + collision_moveable
print(f"Goal reached: {goal_reached}/{NUM_EPISODES} = {goal_reached/NUM_EPISODES}")
print(f"Collision: {collision_count}/{NUM_EPISODES} = {collision_count/NUM_EPISODES}")
print(f"Collision static: {collision_static}/{collision_count} = {collision_static/collision_count}")
print(f"Collision moveable: {collision_moveable}/{collision_count} = {collision_moveable/collision_count}")
print(f"Time is out: {time_is_out}/{NUM_EPISODES} = {time_is_out/NUM_EPISODES}")
print(f"Mean done time: {mean_done_time/goal_reached}")

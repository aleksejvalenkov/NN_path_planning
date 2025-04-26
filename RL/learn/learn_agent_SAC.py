import os
import sys
import numpy as np

import gymnasium as gym
from gymnasium import spaces

# import the skrl components to build the RL system
from skrl.agents.torch.sac import SAC_DEFAULT_CONFIG
from skrl.agents.torch.sac import SAC_RNN as SAC
from skrl.resources.noises.torch import OrnsteinUhlenbeckNoise
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.trainers.torch import ParallelTrainer
from skrl.utils import set_seed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(os.path.dirname(SCRIPT_DIR))


from RL.env.gym_env import CustomEnv
# from RL.agent.SAC_skrl_ResNET_policy import Actor, Critic
from RL.agent.SAC_skrl_RNN_policy import Actor, Critic

# load and wrap the gymnasium environment.
NUM_ENVS = 4
# custom_env = CustomEnv(render_mode="human")
# custom_env = CustomEnv(render_mode=None)

gym.register(id="my_v1",entry_point=CustomEnv, vector_entry_point=CustomEnv)
env = gym.make_vec(id="my_v1", 
                   num_envs=NUM_ENVS, 
                   vectorization_mode="async",
                   seed=None,
                   robot_init_pos=None,
                   robot_goal_pos=None,
                   )

env = wrap_env(env)
device = env.device
print(device)

# instantiate a memory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=20000, num_envs=env.num_envs, device=device, replacement=False)

# instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
models = {}
# models["policy"] = Actor(env.observation_space, env.action_space, device, clip_actions=True)
# models["critic_1"] = Critic(env.observation_space, env.action_space, device)
# models["critic_2"] = Critic(env.observation_space, env.action_space, device)
# models["target_critic_1"] = Critic(env.observation_space, env.action_space, device)
# models["target_critic_2"] = Critic(env.observation_space, env.action_space, device)

models["policy"] = Actor(env.observation_space, env.action_space, device, clip_actions=True, num_envs=env.num_envs)
models["critic_1"] = Critic(env.observation_space, env.action_space, device, num_envs=env.num_envs)
models["critic_2"] = Critic(env.observation_space, env.action_space, device, num_envs=env.num_envs)
models["target_critic_1"] = Critic(env.observation_space, env.action_space, device, num_envs=env.num_envs)
models["target_critic_2"] = Critic(env.observation_space, env.action_space, device, num_envs=env.num_envs)

# initialize models' parameters (weights and biases)
# for model in models.values():
#     model.init_parameters(method_name="normal_", mean=0.0, std=0.1)

cfg = SAC_DEFAULT_CONFIG.copy()
cfg["discount_factor"] = 0.98
cfg["batch_size"] = 100
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 1000
cfg["learn_entropy"] = True
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 200
cfg["experiment"]["checkpoint_interval"] = 25000
cfg["experiment"]["directory"] = "runs/torch/SAC_RNN_adapt_reward"



agent = SAC(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)

# agent.load('runs/torch/SAC_RNN_adapt_reward/25-04-24_09-32-38-637555_SAC_RNN/checkpoints/agent_150000.pt')

# configure and instantiate the RL trainer
# create a sequential trainer
cfg_trainer = {"timesteps": 2000000, "headless": True}
trainer = SequentialTrainer(env=env, agents=[agent], cfg=cfg_trainer)

# start training
trainer.train()

# trainer.eval()
import os
import sys
import numpy as np

import gymnasium as gym
from gymnasium import spaces

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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(os.path.dirname(SCRIPT_DIR))


from PPO.env.gym_env import CustomEnv
from PPO.agent.PPO_skrl_ResNET_policy import Actor, Critic
# from PPO.agent.PPO_skrl_CNN_policy import Actor, Critic
# from PPO.agent.DDPG_skrl_policy import Actor, Critic

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
memory = RandomMemory(memory_size=1024, num_envs=NUM_ENVS, device=device)


# instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
models = {}
models["policy"] = Actor(env.observation_space, env.action_space, device, clip_actions=True)
models["value"] = Critic(env.observation_space, env.action_space, device)


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 4024  # memory_size
cfg["learning_epochs"] = 10
cfg["mini_batches"] = 64
cfg["discount_factor"] = 0.9
cfg["lambda"] = 0.95
cfg["learning_rate"] = 1e-3
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
cfg["experiment"]["write_interval"] = 200
cfg["experiment"]["checkpoint_interval"] = 20000
cfg["experiment"]["directory"] = "runs/torch/PPO_RESNET_norm_reward_gen2"


agent = PPO(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)


# agent.load('runs/torch/PPO_with_A_withback_26obs_2acts_angle_window/25-03-29_22-30-10-695238_PPO/checkpoints/agent_50000.pt')

# configure and instantiate the RL trainer
# create a sequential trainer
cfg_trainer = {"timesteps": 2000000, "headless": True}
trainer = SequentialTrainer(env=env, agents=[agent], cfg=cfg_trainer)

# start training
trainer.train()

# trainer.eval()
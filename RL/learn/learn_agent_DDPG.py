import os
import sys
import numpy as np

import gymnasium as gym
from gymnasium import spaces

# import the skrl components to build the RL system
from skrl.agents.torch.ddpg import DDPG, DDPG_DEFAULT_CONFIG
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
from RL.agent.DDPG_skrl_ResNET_policy import Actor, Critic

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
memory = RandomMemory(memory_size=2000, num_envs=NUM_ENVS, device=device)


# instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
models = {}
models["policy"] = Actor(env.observation_space, env.action_space, device)
models["target_policy"] = Actor(env.observation_space, env.action_space, device)
models["critic"] = Critic(env.observation_space, env.action_space, device)
models["target_critic"] = Critic(env.observation_space, env.action_space, device)


cfg = DDPG_DEFAULT_CONFIG.copy()
cfg["exploration"]["noise"] = OrnsteinUhlenbeckNoise(theta=0.15, sigma=0.1, base_scale=1.0, device=device)
cfg["batch_size"] = 100
cfg["random_timesteps"] = 100
cfg["learning_starts"] = 100
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 100
cfg["experiment"]["checkpoint_interval"] = 25000
cfg["experiment"]["directory"] = "runs/torch/DDPG_noback"


# agent = PPO(models=models,
#             memory=memory,
#             cfg=cfg,
#             observation_space=env.observation_space,
#             action_space=env.action_space,
#             device=device)

agent = DDPG(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)

agent.load('runs/torch/DDPG_noback/25-04-13_12-08-26-293962_DDPG/checkpoints/agent_150000.pt')

# configure and instantiate the RL trainer
# create a sequential trainer
cfg_trainer = {"timesteps": 2000000, "headless": True}
trainer = SequentialTrainer(env=env, agents=[agent], cfg=cfg_trainer)

# start training
# trainer.train()

trainer.eval()
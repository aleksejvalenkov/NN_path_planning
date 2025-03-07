import os
import numpy as np

import gymnasium as gym
from gymnasium import spaces

import torch
import torch.nn as nn
import torch.nn.functional as F

# import the skrl components to build the RL system
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed


from Simulator.gui.sim import Simulator




class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        self.sim_env = Simulator()
        # Example when using discrete actions:
        self.action_space = spaces.Box(low=-1, high=1,
                                            shape=(3,), dtype=np.float64)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=-3, high=3,
                                            shape=(27,), dtype=np.float64)
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        if render_mode is not None:
            self.sim_env.init_window()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.sim_env.step(action)
        self.render()
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        observation, info = self.sim_env.reset()
        return observation, info

    def render(self):
        self.sim_env.render()

    def close(self):
        self.sim_env.kill_window()

# seed for reproducibility
set_seed()  # e.g. `set_seed(42)` for fixed seed

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # self._init_weights()

    def forward(self, x):
        residual = x
        out = F.leaky_relu(self.fc1(x))
        out = self.fc2(out)
        out += residual
        return F.leaky_relu(out)

    # def _init_weights(self):
    #     torch.nn.init.xavier_uniform_(self.fc1.weight)
    #     torch.nn.init.xavier_uniform_(self.fc2.weight)


# define the model
class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device,
                 clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        state_dim = int(self.num_observations)
        res1_out_dim = state_dim
        res2_out_dim = state_dim * 2
        conc1_dim = state_dim + res1_out_dim

        self.res_block1 = ResidualBlock(
            input_dim=state_dim,
            hidden_dim=512,
            output_dim=res1_out_dim,
        )
        self.res_block2 = ResidualBlock(
            input_dim=conc1_dim,
            hidden_dim=512,
            output_dim=res2_out_dim,
        )
        self.fc = nn.Linear(res2_out_dim + conc1_dim, int(self.num_actions))
        # torch.nn.init.xavier_uniform_(self.fc.weight)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        state = inputs["states"]
        x = self.res_block1(state)
        x = torch.cat([x, state], dim=-1)

        concat1 = x

        x = self.res_block2(x)
        concat2 = torch.cat([x, concat1], dim=-1)

        output = F.tanh(self.fc(concat2))

        return output, self.log_std_parameter, {}


class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        state_dim = int(self.num_observations)
        res1_out_dim = state_dim
        res2_out_dim = state_dim * 2
        conc1_dim = state_dim + res1_out_dim

        self.res_block1 = ResidualBlock(
            input_dim=state_dim,
            hidden_dim=512,
            output_dim=res1_out_dim,
        )
        self.res_block2 = ResidualBlock(
            input_dim=conc1_dim,
            hidden_dim=512,
            output_dim=res2_out_dim,
        )
        self.fc = nn.Linear(res2_out_dim + conc1_dim, int(1))
        # torch.nn.init.xavier_uniform_(self.fc.weight)

    def compute(self, inputs, role):
        state = inputs["states"]
        x = self.res_block1(state)
        x = torch.cat([x, state], dim=-1)

        concat1 = x

        x = self.res_block2(x)
        concat2 = torch.cat([x, concat1], dim=-1)

        output = F.tanh(self.fc(concat2))

        return output, {}


# load and wrap the gymnasium environment.

env = CustomEnv(render_mode="human")


# note: the environment version may change depending on the gymnasium version
# try:
#     env = gym.make_vec("Pendulum-v1", num_envs=4, vectorization_mode="sync")
# except (gym.error.DeprecatedEnv, gym.error.VersionNotFound) as e:
#     env_id = [spec for spec in gym.envs.registry if spec.startswith("Pendulum-v")][0]
#     print("Pendulum-v1 not found. Trying {}".format(env_id))
#     env = gym.make_vec(env_id, num_envs=4, vectorization_mode="sync")
# env.device = "cpu"
env = wrap_env(env)
device = env.device


# instantiate a memory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=2048, device=device)


# instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
models = {}
models["policy"] = Policy(env.observation_space, env.action_space, device, clip_actions=True)
models["value"] = Value(env.observation_space, env.action_space, device)


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 2048  # memory_size
cfg["learning_epochs"] = 10
cfg["mini_batches"] = 32
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
cfg["experiment"]["write_interval"] = 500
cfg["experiment"]["checkpoint_interval"] = 50000
cfg["experiment"]["directory"] = "runs/torch/robot"

agent = PPO(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)

# agent.load('runs/torch/robot/25-03-07_22-34-23-484590_PPO/checkpoints/best_agent.pt')

# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 1000000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])

# start training
trainer.train()

# trainer.eval()
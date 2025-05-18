import os
import sys
import numpy as np
import torch
# import the skrl components to build the RL system
from skrl.agents.torch.rpo import RPO, RPO_DEFAULT_CONFIG
# from skrl.agents.torch.rpo import RPO_RNN as RPO, RPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(os.path.dirname(SCRIPT_DIR))

from RL.env.gym_env import CustomEnv
# from RL.agent.PPO_skrl_RNN_policy import Actor, Critic
from RL.agent.PPO_skrl_ResNET_policy import Actor, Critic



class Agent:
    def __init__(self, observation_space, action_space, device='cuda:0'):
        self.device = device
        memory = RandomMemory(memory_size=1024, device=device)
        models = {}
        models["policy"] = Actor(observation_space, action_space, device)
        models["value"] = Critic(observation_space, action_space, device)

        # configure and instantiate the agent (visit its documentation to see all the options)
        cfg = RPO_DEFAULT_CONFIG.copy()
        cfg["rollouts"] = 1024  # memory_size
        cfg["learning_epochs"] = 10
        cfg["mini_batches"] = 32
        cfg["alpha"] =  0.1 
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
        cfg["state_preprocessor"] = None
        cfg["state_preprocessor_kwargs"] = {}
        cfg["value_preprocessor"] = None
        cfg["value_preprocessor_kwargs"] = {}
        # logging to TensorBoard and write checkpoints (in timesteps)
        cfg["experiment"]["write_interval"] = 500
        cfg["experiment"]["checkpoint_interval"] = 50000
        cfg["experiment"]["directory"] = "runs/torch/metric_env_and_ansiolute_penalty_evaluate"

        self.agent = RPO(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=observation_space,
            action_space=action_space,
            device=self.device)
        
        # self.agent.load('runs/torch/RPO_LSTM_base_reward_angle_focus/25-04-28_21-59-13-384451_RPO_RNN/checkpoints/best_agent.pt')
        # self.agent.load('runs/torch/RPO_LSTM_base_reward_angle_focus/25-04-28_21-59-13-384451_RPO_RNN/checkpoints/best_agent.pt')
        
    def gen_action(self, observation):
        # observation = torch.tensor(observation["state"], dtype=torch.float32, device=self.device)
        observation = np.zeros((27), dtype=np.float32)
        observation = torch.tensor(observation, dtype=torch.float32, device=self.device)
        # state = {"states": observation, "rnn": [torch.zeros(1, 1, 256).to(self.device), torch.zeros(1, 1, 256).to(self.device)]}
        # print("states = ", observation)
        action = self.agent.act(observation, timestep=0, timesteps=1)
        action_np = action[0].cpu().detach().numpy()
        return action_np
    
class AgentSafe:
    def __init__(self, observation_space, action_space, device='cuda:0'):
        self.device = device
        memory = RandomMemory(memory_size=1024, device=device)
        models = {}
        models["policy"] = Actor(observation_space, action_space, device, num_envs=1)
        models["value"] = Critic(observation_space, action_space, device, num_envs=1)

        # configure and instantiate the agent (visit its documentation to see all the options)
        cfg = RPO_DEFAULT_CONFIG.copy()
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
        cfg["state_preprocessor_kwargs"] = {"size": observation_space, "device": self.device}
        cfg["value_preprocessor"] = RunningStandardScaler
        cfg["value_preprocessor_kwargs"] = {"size": 1, "device": self.device}
        # logging to TensorBoard and write checkpoints (in timesteps)
        cfg["experiment"]["write_interval"] = 500
        cfg["experiment"]["checkpoint_interval"] = 50000
        cfg["experiment"]["directory"] = "runs/torch/metric_env_and_ansiolute_penalty_evaluate"

        self.agent = RPO(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=observation_space,
            action_space=action_space,
            device=self.device)
        
        # self.agent.load('runs/torch/RPO_LSTM_base_reward_angle_focus/25-04-28_21-59-13-384451_RPO_RNN/checkpoints/best_agent.pt')
        self.agent.load('runs/torch/RPO_LSTM_base_reward_angle_focus/25-04-28_21-59-13-384451_RPO_RNN/checkpoints/best_agent.pt')
        
    def gen_action(self, observation):
        observation = torch.tensor(observation, dtype=torch.float32, device=self.device)
        action = self.agent.act(observation, timestep=0, timesteps=1)
        action_np = action[0].cpu().detach().numpy()
        return action_np
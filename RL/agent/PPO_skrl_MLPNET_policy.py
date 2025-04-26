import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

# define the model
class Actor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device,
                 clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.fc1 = nn.Linear(self.num_observations, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 32)
        self.fc4 = nn.Linear(32, self.num_actions)


        # torch.nn.init.xavier_uniform_(self.fc.weight)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        state = inputs["states"]
        # print("state net = ", state)

        x = self.fc1(state)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        x = self.fc3(x)
        x = F.leaky_relu(x)
        x = self.fc4(x)

        output = F.tanh(x)

        # perturb the mean actions by adding a randomized uniform sample
        # rpo_alpha = inputs["alpha"]
        # perturbation = torch.zeros_like(x).uniform_(-rpo_alpha, rpo_alpha)
        # x += perturbation

        # print('output = ', output)
        return output, self.log_std_parameter, {}


class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.fc1 = nn.Linear(self.num_observations, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 32)
        self.fc4 = nn.Linear(32, 1)

        # torch.nn.init.xavier_uniform_(self.fc.weight)

    def compute(self, inputs, role):
        state = inputs["states"]
        # print("state net = ", state)
        x = self.fc1(state)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        x = self.fc3(x)
        x = F.leaky_relu(x)
        x = self.fc4(x)

        output = x
        # print("output value = ", output)

        return output, {}
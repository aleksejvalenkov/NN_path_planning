import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.models.torch import DeterministicMixin, GaussianMixin, CategoricalMixin, Model

class Actor(CategoricalMixin, Model):
    def __init__(self, observation_space, action_space, device, unnormalized_log_prob=True):
        Model.__init__(self, observation_space, action_space, device)
        CategoricalMixin.__init__(self, unnormalized_log_prob)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=2, stride=1)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512) 
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, self.num_actions)

        # self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        # permute (samples, width * height * channels) -> (samples, channels, width, height)
        # Reshape the input states to match the expected dimensions for the convolutional layers
        # (samples, width * height * channels) -> (samples, channels, width, height)
        state = inputs["states"].view(-1, *self.observation_space.shape).permute(0, 3, 1, 2)
        # print('state = ', state)
        # print('polocy', state.shape)
        x = self.conv1(state)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = F.relu(x)

        # print(x.shape)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = F.relu(x)

        x = self.fc4(x)
        # print(x)
        output = x
        # if len(state.shape) > 3:
        #     output_vx = output[:,:1]
        #     output_w = output[:,1:2]
        # else:
        #     output_vx = output[:1]
        #     output_w = output[1:2]
        print('output = ', output)
        # # print('output 2 = ', output[:,1:2])
        # output_vec = torch.stack([F.sigmoid(output_vx), F.tanh(output_w)], dim=-1).reshape(output.size())
        return output, {}


class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=2, stride=1)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512) 
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 1)

    def compute(self, inputs, role):
        # permute (samples, width * height * channels) -> (samples, channels, width, height)
        state = inputs["states"].view(-1, *self.observation_space.shape).permute(0, 3, 1, 2)
        x = self.conv1(state)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = F.relu(x)

        # print(x.shape)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = F.relu(x)

        x = self.fc4(x)
        return x, {}

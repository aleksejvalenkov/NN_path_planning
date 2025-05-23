import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

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
class Actor(DeterministicMixin, Model):
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
        self.fc = nn.Linear(res2_out_dim + conc1_dim, int(self.num_actions))

        # torch.nn.init.xavier_uniform_(self.fc.weight)
        # self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        state = inputs["states"]

        x = self.res_block1(state)
        x = torch.cat([x, state], dim=-1)

        concat1 = x

        x = self.res_block2(x)
        concat2 = torch.cat([x, concat1], dim=-1)

        output = F.tanh(self.fc(concat2))


        return output, {}


class Critic(DeterministicMixin, Model):
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
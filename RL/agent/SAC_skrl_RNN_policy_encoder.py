import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

# define the model
class Actor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum",
                 num_envs=1, num_layers=1, hidden_size=256, sequence_length=20):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.num_envs = num_envs
        self.num_layers = num_layers
        self.hidden_size = hidden_size  # Hout
        self.sequence_length = sequence_length

        # self.con_1 = nn.Conv1d(1, 8, kernel_size=3, stride=2, padding=1)
        # self.con_2 = nn.Conv1d(8, 16, kernel_size=3, stride=2, padding=1)
        # self.con_3 = nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1)

        self.fc_lidar_1 = nn.Linear(180, 64)
        self.fc_lidar_2 = nn.Linear(64, 32)
        self.fc_vel_1 = nn.Linear(3, 32)
        self.fc_goal_1 = nn.Linear(4, 32)

        self.lstm = nn.LSTM(input_size=96,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True)  # batch_first -> (batch, sequence, features)

        self.net = nn.Sequential(nn.Linear(self.hidden_size, 256),
                                 nn.LeakyReLU(),
                                 nn.Linear(256, 128),
                                 nn.LeakyReLU(),
                                 nn.Linear(128, 64),
                                 nn.LeakyReLU(),
                                 nn.Linear(64, self.num_actions))

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def get_specification(self):
        # batch size (N) is the number of envs
        return {"rnn": {"sequence_length": self.sequence_length,
                        "sizes": [(self.num_layers, self.num_envs, self.hidden_size),    # hidden states (D ∗ num_layers, N, Hout)
                                  (self.num_layers, self.num_envs, self.hidden_size)]}}  # cell states   (D ∗ num_layers, N, Hcell)

    def compute(self, inputs, role):
        # print("inputs = ", inputs)
        states_input = inputs["states"]
        terminated = inputs.get("terminated", None)
        hidden_states, cell_states = inputs["rnn"][0], inputs["rnn"][1]

        lidar = states_input[:, 0:180]
        lidar = self.fc_lidar_1(lidar)
        lidar = F.relu(lidar)
        lidar = self.fc_lidar_2(lidar)
        lidar = F.relu(lidar)

        # lidar = lidar.unsqueeze(1) 
        # lidar = self.con_1(lidar)
        # lidar = F.relu(lidar)
        # lidar = self.con_2(lidar)
        # lidar = F.relu(lidar)
        # lidar = self.con_3(lidar)
        # lidar = F.relu(lidar)
        # lidar = lidar.view(states_input.size(0), -1)  # Ensure lidar matches batch size
        # print("lidar shape = ", lidar.shape)

        # Ensure all tensors have the same size along dimension 1
        vel = states_input[:, 180:183]
        vel = F.relu(self.fc_vel_1(vel))
        # vel = vel.view(states_input.size(0), -1)  # Ensure vel matches batch size
        # print("vel shape = ", vel.shape)

        goal = states_input[:, 183:187]
        goal = F.relu(self.fc_goal_1(goal))
        # goal = goal.view(states_input.size(0), -1)  # Ensure goal matches batch size
        # print("goal shape = ", goal.shape)

        states = torch.cat((lidar, vel, goal), dim=1)

        # training
        if self.training:
            rnn_input = states.view(-1, self.sequence_length, states.shape[-1])  # (N, L, Hin): N=batch_size, L=sequence_length

            hidden_states = hidden_states.view(self.num_layers, -1, self.sequence_length, hidden_states.shape[-1])  # (D * num_layers, N, L, Hout)
            cell_states = cell_states.view(self.num_layers, -1, self.sequence_length, cell_states.shape[-1])  # (D * num_layers, N, L, Hcell)
            # get the hidden/cell states corresponding to the initial sequence
            hidden_states = hidden_states[:,:,0,:].contiguous()  # (D * num_layers, N, Hout)
            cell_states = cell_states[:,:,0,:].contiguous()  # (D * num_layers, N, Hcell)

            # reset the RNN state in the middle of a sequence
            if terminated is not None and torch.any(terminated):
                rnn_outputs = []
                terminated = terminated.view(-1, self.sequence_length)
                indexes = [0] + (terminated[:,:-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist() + [self.sequence_length]

                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    rnn_output, (hidden_states, cell_states) = self.lstm(rnn_input[:,i0:i1,:], (hidden_states, cell_states))
                    hidden_states[:, (terminated[:,i1-1]), :] = 0
                    cell_states[:, (terminated[:,i1-1]), :] = 0
                    rnn_outputs.append(rnn_output)

                rnn_states = (hidden_states, cell_states)
                rnn_output = torch.cat(rnn_outputs, dim=1)
            # no need to reset the RNN state in the sequence
            else:
                rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))
        # rollout
        else:
            rnn_input = states.view(-1, 1, states.shape[-1])  # (N, L, Hin): N=num_envs, L=1
            rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))

        # flatten the RNN output
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)  # (N, L, D ∗ Hout) -> (N * L, D ∗ Hout)

        return torch.tanh(self.net(rnn_output)), self.log_std_parameter, {"rnn": [rnn_states[0], rnn_states[1]]}

class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 num_envs=1, num_layers=1, hidden_size=256, sequence_length=20):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.num_envs = num_envs
        self.num_layers = num_layers
        self.hidden_size = hidden_size  # Hout
        self.sequence_length = sequence_length

        # self.con_1 = nn.Conv1d(1, 8, kernel_size=3, stride=2, padding=1)
        # self.con_2 = nn.Conv1d(8, 16, kernel_size=3, stride=2, padding=1)
        # self.con_3 = nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1)

        self.fc_lidar_1 = nn.Linear(180, 64)
        self.fc_lidar_2 = nn.Linear(64, 32)
        self.fc_vel_1 = nn.Linear(3, 32)
        self.fc_goal_1 = nn.Linear(4, 32)

        self.lstm = nn.LSTM(input_size=96,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True)  # batch_first -> (batch, sequence, features)

        self.net = nn.Sequential(nn.Linear(self.hidden_size + self.num_actions, 256),
                                 nn.LeakyReLU(),
                                 nn.Linear(256, 128),
                                 nn.LeakyReLU(),
                                 nn.Linear(128, 64),
                                 nn.LeakyReLU(),
                                 nn.Linear(64, 1))

    def get_specification(self):
        # batch size (N) is the number of envs
        return {"rnn": {"sequence_length": self.sequence_length,
                        "sizes": [(self.num_layers, self.num_envs, self.hidden_size),    # hidden states (D ∗ num_layers, N, Hout)
                                  (self.num_layers, self.num_envs, self.hidden_size)]}}  # cell states   (D ∗ num_layers, N, Hcell)

    def compute(self, inputs, role):
        states_input = inputs["states"]
        terminated = inputs.get("terminated", None)
        hidden_states, cell_states = inputs["rnn"][0], inputs["rnn"][1]

        lidar = states_input[:, 0:180]

        lidar = self.fc_lidar_1(lidar)
        lidar = F.relu(lidar)
        lidar = self.fc_lidar_2(lidar)
        lidar = F.relu(lidar)

        # lidar = lidar.unsqueeze(1) 
        # lidar = self.con_1(lidar)
        # lidar = F.relu(lidar)
        # lidar = self.con_2(lidar)
        # lidar = F.relu(lidar)
        # lidar = self.con_3(lidar)
        # lidar = F.relu(lidar)
        # lidar = lidar.view(states_input.size(0), -1)  # Ensure lidar matches batch size
        # print("lidar shape = ", lidar.shape)

        vel = states_input[:, 180:183]
        vel = F.relu(self.fc_vel_1(vel))
        goal = states_input[:, 183:187]
        goal = F.relu(self.fc_goal_1(goal))
        states = torch.cat((lidar, vel, goal), dim=1)

        # training
        if self.training:
            rnn_input = states.view(-1, self.sequence_length, states.shape[-1])  # (N, L, Hin): N=batch_size, L=sequence_length

            hidden_states = hidden_states.view(self.num_layers, -1, self.sequence_length, hidden_states.shape[-1])  # (D * num_layers, N, L, Hout)
            cell_states = cell_states.view(self.num_layers, -1, self.sequence_length, cell_states.shape[-1])  # (D * num_layers, N, L, Hcell)
            # get the hidden/cell states corresponding to the initial sequence
            hidden_states = hidden_states[:,:,0,:].contiguous()  # (D * num_layers, N, Hout)
            cell_states = cell_states[:,:,0,:].contiguous()  # (D * num_layers, N, Hcell)

            # reset the RNN state in the middle of a sequence
            if terminated is not None and torch.any(terminated):
                rnn_outputs = []
                terminated = terminated.view(-1, self.sequence_length)
                indexes = [0] + (terminated[:,:-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist() + [self.sequence_length]

                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    rnn_output, (hidden_states, cell_states) = self.lstm(rnn_input[:,i0:i1,:], (hidden_states, cell_states))
                    hidden_states[:, (terminated[:,i1-1]), :] = 0
                    cell_states[:, (terminated[:,i1-1]), :] = 0
                    rnn_outputs.append(rnn_output)

                rnn_states = (hidden_states, cell_states)
                rnn_output = torch.cat(rnn_outputs, dim=1)
            # no need to reset the RNN state in the sequence
            else:
                rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))
        # rollout
        else:
            rnn_input = states.view(-1, 1, states.shape[-1])  # (N, L, Hin): N=num_envs, L=1
            rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))

        # flatten the RNN output
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)  # (N, L, D ∗ Hout) -> (N * L, D ∗ Hout)

        return self.net(torch.cat([rnn_output, inputs["taken_actions"]], dim=1)), {"rnn": [rnn_states[0], rnn_states[1]]}
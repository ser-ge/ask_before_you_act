#!/usr/bin/env python3

import torch

import torch.nn as nn

device = 'cpu'

class BaselineModel(nn.Module):
    def __init__(self, action_dim=7,mem_hidden_dim=64):
        super().__init__()

        self.cnn_encoding_dim = 64

        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, self.cnn_encoding_dim, (2, 2)),
            nn.ReLU())

        self.mem_hidden_dim = mem_hidden_dim # only used in init memory to pass to agent to act
        # but not actually passed in the pure baseline case
        self.policy_input_dim = self.cnn_encoding_dim
        self.policy_head = nn.Linear(self.policy_input_dim, action_dim)
        self.value_head = nn.Linear(self.policy_input_dim, 1)
        self.activation = nn.ReLU()

    def policy(self, obs):
        x = self.encode_obs(obs)
        action_policy = self.policy_head(x)
        return action_policy

    def value(self, obs, hist_mem):
        x = self.encode_obs(obs)
        state_value = self.value_head(x)
        return state_value

    def encode_obs(self, obs):
        x = obs.view(-1, 3, 7, 7)  # x: (batch, C_in, H_in, W_in)
        obs_encoding = self.image_conv(x).view(-1, self.cnn_encoding_dim)  # x: (batch, hidden)
        return obs_encoding

class BaselineModelExpMem(BaselineModel):
    def __init__(self, action_dim=7, mem_hidden_dim=64):
        super().__init__(action_dim)

        self.mem_hidden_dim = mem_hidden_dim
        self.policy_input_dim = self.cnn_encoding_dim + self.mem_hidden_dim

        self.memory_rnn = nn.LSTMCell(self.cnn_encoding_dim + action_dim,
                                      self.mem_hidden_dim)

        self.policy_head = nn.Linear(self.policy_input_dim, action_dim)
        self.value_head = nn.Linear(self.policy_input_dim, 1)

    def policy(self, obs, hist_mem):
        x = self.encode_obs(obs)
        x = torch.cat((x, hist_mem), 1)
        action_policy = self.policy_head(x)
        return action_policy

    def value(self, obs, hist_mem):
        x = self.encode_obs(obs)
        x = torch.cat((x, hist_mem), 1)
        state_value = self.value_head(x)
        return state_value

    def remember(self, obs, action, memory):
        encoded_obs = self.encode_obs(obs)
        x = torch.cat((encoded_obs, action), 1)
        return self.memory_rnn(x, memory)

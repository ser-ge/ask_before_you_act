#!/usr/bin/env python3

import torch.nn as nn


class BaselineCNN(nn.Module):
    def __init__(self, action_dim=7):
        super().__init__()

        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU())

        self.policy_head = nn.Linear(64, action_dim)
        self.value_head = nn.Linear(64, 1)
        self.activation = nn.ReLU()

    def forward(self, x, flag="policy"):
        # Shared Body
        x = x.view(-1, 3, 7, 7)  # x: (batch, C_in, H_in, W_in)
        x = self.image_conv(x).squeeze()  # x: (batch, hidden)
        # Split heads
        if flag == "policy":
            x_pol = self.policy_head(x)
            return x_pol
        elif flag == "value":
            x_val = self.value_head(x)
            return x_val


class BaselineMem(BaselineCNN):  # TODO - check memory for baseline
    def __init__(self, action_dim=7):
        super().__init__(action_dim)
        self.memory_rnn = nn.LSTMCell(7 + 64, 64)

    def remember(self, obs, action, memory):
        encoded_obs = self.encode_obs(obs)
        x = torch.cat((encoded_obs, action), 1)
        return self.memory_rnn(x, memory)


#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.distributions as distributions
from language_model.model import Model as QuestionRNN

QUESTION_SAMPLING_TEMP = 0.9


class BrainNet(nn.Module):
    def __init__(self, question_rnn, action_dim=7, vocab_size=10):
        super().__init__()

        self.vocab_size = vocab_size

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

        # CNN output is 64 dims
        # Assuming qa_history is also 64
        # self.question_rnn = nn.LSTMCell(self.vocab_size, 128)  # (input_size, hidden_size)
        # self.question_head = nn.Linear(128, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def policy(self, obs, answer, word_lstm_hidden):
        """
        word_lstm_hidden : last hidden state
        """
        encoded_obs = self.encode_obs(obs)
        # x = torch.cat((encoded_obs, answer, word_lstm_hidden), 1)
        action_policy = self.policy_head(encoded_obs)
        return action_policy

    def value(self, obs, answer, word_lstm_hidden):
        encoded_obs = self.encode_obs(obs)
        # x = torch.cat((encoded_obs, answer, word_lstm_hidden), 1)
        state_value = self.value_head(encoded_obs)
        return state_value

    def encode_obs(self, obs):
        x = obs.view(-1, 3, 7, 7)  # x: (batch, C_in, H_in, W_in)
        # batch = x.shape[0]
        obs_encoding = self.image_conv(x).view(-1, 64)  # x: (batch, hidden)
        return obs_encoding


#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)

class SharedCNN(nn.Module):
    def __init__(self, action_dim=7):
        super(SharedCNN, self).__init__()

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


class PPOAgent():
    def __init__(self, state_dim, action_dim, hidden_dim=64, learning_rate=0.001,
                 gamma=0.99, clip_param=0.2, value_param=1, entropy_param=0.01,
                 lmbda=0.95, backward_epochs=2):
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.lmbda = lmbda

        self.model = SharedCNN().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.clip_param = clip_param
        self.entropy_param = entropy_param
        self.value_param = value_param
        self.episode = 0
        self.T = 1

        self.done = True
        self.data = []
        self.backward_epochs = backward_epochs

    def act(self, observation, exploration=True):
        # Calculate policy
        observation = torch.FloatTensor(observation).to(device)

        logits = self.model(observation, flag="policy")
        action_prob = F.softmax(logits.squeeze() / self.T, dim=-1)
        dist = distributions.Categorical(action_prob)
        action = dist.sample()
        probs = action_prob[action]  # Policy log prob
        entropy = dist.entropy()  # Entropy regularizer
        return action.detach().item(), probs, entropy

    def update(self):
        obs, a, reward, next_obs, done, log_prob, entropy = self.get_batch()

        for i in range(self.backward_epochs):
            # Get current V

            V_pred = self.model(obs, flag="value").squeeze()

            # Get next V
            next_V_pred = self.model(next_obs, flag="value").squeeze()

            # Compute TD error
            target = reward.squeeze().to(device) + self.gamma * next_V_pred * done.squeeze().to(device)
            td_error = (target - V_pred).detach()

            # Generalised Advantage Estimation
            advantage_list = []
            advantage = 0.0
            for delta in reversed(td_error):
                advantage = self.gamma * self.lmbda * advantage + delta
                advantage_list.append([advantage])
            advantage_list.reverse()
            advantage = torch.FloatTensor(advantage_list).to(device)

            # Clipped PPO Policy Loss
            logits = self.model(obs, flag="policy")
            probs = F.softmax(logits, dim=-1)
            pi_a = probs.squeeze(1).gather(1, torch.LongTensor(a.long()).to(device))
            ratio = torch.exp(torch.log(pi_a) - torch.log(log_prob))
            surrogate1 = ratio * advantage
            surrogate2 = advantage * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            L_clip = torch.min(surrogate1, surrogate2).mean()

            # Entropy regularizer
            L_entropy = self.entropy_param * entropy.detach().mean()

            # Value function loss
            L_value = self.value_param * F.smooth_l1_loss(V_pred, target.detach())

            total_loss = -(L_clip - L_value + L_entropy).to(device)

            # Update params
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
        return total_loss.item()

    def store(self, transition):
        self.data.append(transition)

    def get_batch(self):
        obs_batch = []
        a_batch = []
        r_batch = []
        next_obs_batch = []
        log_prob_batch = []
        entropy_batch = []
        done_batch = []
        for transition in self.data:
            obs, a, r, next_obs, log_prob, entropy, done = transition

            obs_batch.append(obs)
            a_batch.append([a])
            r_batch.append([r])
            next_obs_batch.append(next_obs)
            log_prob_batch.append([log_prob])
            entropy_batch.append([entropy])
            done_bool = 0 if done else 1
            done_batch.append([done_bool])

        obs = torch.FloatTensor(obs_batch).to(device)
        a = torch.FloatTensor(a_batch).to(device)
        r = torch.FloatTensor(r_batch).to(device)
        next_obs = torch.FloatTensor(next_obs_batch).to(device)
        log_prob = torch.FloatTensor(log_prob_batch).to(device)
        entropy = torch.FloatTensor(entropy_batch).to(device)
        done = torch.FloatTensor(done_batch).to(device)

        self.data = []

        return obs, a, r, next_obs, done, log_prob, entropy

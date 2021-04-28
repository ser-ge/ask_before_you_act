#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions

device = "cpu"


class PPOAgent:
    def __init__(self, model, learning_rate=0.001, lmbda=0.95, gamma=0.99,
                 clip_param=0.2, value_param=1, entropy_act_param=0.01):

        self.gamma = gamma
        self.lmbda = lmbda

        self.model = model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.clip_param = clip_param
        self.entropy_param = entropy_act_param
        self.value_param = value_param
        self.episode = 0
        self.T = 1

        self.done = True
        self.data = []

    def act(self, observation):
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


class PPOAgentMem(PPOAgent):  # TODO - check memory baseline agent
    def __init__(self, model, learning_rate=0.001, lmbda=0.95, gamma=0.99,
                 clip_param=0.2, value_param=1, entropy_act_param=0.01):
        super().__init__(model, learning_rate, lmbda, gamma,
                         clip_param, value_param, entropy_act_param)

    def remember(self, state, action, memory):
        action_one_hot = torch.zeros((1, 7)).to(device)
        action_one_hot[0, action] = 1
        obs = torch.FloatTensor(state).to(device)
        mem = self.model.remember(obs, action_one_hot, memory)
        return mem


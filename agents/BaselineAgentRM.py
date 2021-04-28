#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
from utils.Trainer import TransitionNoQA

device = "cpu"

class BaselineAgent:
    def __init__(self, model, learning_rate=0.001, lmbda=0.95, gamma=0.99,
                 clip_param=0.2, value_param=1, entropy_act_param=0.01):

        self.gamma = gamma
        self.lmbda = lmbda
        self.baseline = True

        self.model = model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.clip_param = clip_param
        self.entropy_act_param = entropy_act_param
        self.value_param = value_param
        self.episode = 0
        self.T = 1

        self.done = True
        self.data = []

    def act(self, observation, hist_mem):
        # Calculate policy

        observation = torch.FloatTensor(observation).to(device)
        logits = self.model.policy(observation,hist_mem)
        action_prob = F.softmax(logits.squeeze() / self.T, dim=-1)
        dist = distributions.Categorical(action_prob)
        action = dist.sample()
        probs = action_prob[action]  # Policy log prob
        entropy = dist.entropy()  # Entropy regularizer
        return action.detach().item(), probs, entropy

    def update(self):
        state, action, reward, next_state, \
        log_prob_act, entropy_act, done, hidden_hist_mem, \
        cell_hist_mem, next_hidden_hist_mem, next_cell_hist_mem = self.get_batch()

        # Get current V
        V_pred = self.model.value(state,hidden_hist_mem).squeeze()

        # Get next V
        next_V_pred = self.model.value(next_state,hidden_hist_mem).squeeze()

        # Compute TD error
        target = reward.squeeze().to(device) + self.gamma * next_V_pred * done.squeeze().to(device)
        td_error = (target - V_pred).detach()

        # Generalised Advantage Estimation
        advantage = self.gae(td_error)

        # Clipped PPO Policy Loss
        L_clip = self.clip_loss(action, advantage, log_prob_act, state,hidden_hist_mem)

        # Entropy regularizer
        L_entropy = self.entropy_act_param * entropy_act.detach().mean()

        # Value function loss
        L_value = self.value_param * F.smooth_l1_loss(V_pred, target.detach())

        # Total loss
        total_loss = -(L_clip - L_value + L_entropy).to(device)

        # Update paramss
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()

    def gae(self, td_error):
        advantage_list = []
        advantage = 0.0
        for delta in reversed(td_error):
            advantage = self.gamma * self.lmbda * advantage + delta
            advantage_list.append([advantage])
        advantage_list.reverse()
        advantage = torch.FloatTensor(advantage_list).to(device)
        return advantage

    def clip_loss(self, action, advantage, log_prob_act, state,hidden_hist):
        logits = self.model.policy(state,hidden_hist)
        probs = F.softmax(logits, dim=-1)
        pi_a = probs.squeeze(1).gather(1, action.long())
        ratio = torch.exp(torch.log(pi_a) - torch.log(log_prob_act))
        surrogate1 = ratio * advantage
        surrogate2 = advantage * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
        L_clip = torch.min(surrogate1, surrogate2).mean()
        return L_clip

    def get_batch(self):
        trans = TransitionNoQA(*zip(*self.data))

        state = torch.FloatTensor(trans.state).to(device)
        action = torch.FloatTensor(trans.action).to(device).view(-1, 1)
        reward = torch.FloatTensor(trans.reward).to(device).view(-1, 1)
        next_state = torch.FloatTensor(trans.next_state).to(device)
        log_prob_act = torch.FloatTensor(trans.log_prob_act).to(device).view(-1, 1)
        entropy_act = torch.FloatTensor(trans.entropy_act).to(device).view(-1, 1)
        done = ~torch.BoolTensor(trans.done).to(device).view(-1, 1)  # You need the tilde!
        hidden_hist_mem = torch.cat(trans.hidden_hist_mem)
        cell_hist_mem = torch.cat(trans.cell_hist_mem)
        next_hidden_hist_mem = torch.cat(trans.next_hidden_hist_mem)
        next_cell_hist_mem = torch.cat(trans.next_cell_hist_mem)

        self.data = []

        return (state, action, reward,
                next_state, log_prob_act, entropy_act,
                done,  hidden_hist_mem, cell_hist_mem, next_hidden_hist_mem,
                next_cell_hist_mem)

    def store(self, transition):
        self.data.append(transition)

    def init_memory(self):
        return (torch.rand(1, self.model.mem_hidden_dim),
                torch.rand(1, self.model.mem_hidden_dim))


class BaselineAgentMem(BaselineAgent):
    def __init__(self, model, learning_rate=0.001, lmbda=0.95, gamma=0.99,
                 clip_param=0.2, value_param=1, entropy_act_param=0.01):
        super().__init__(model,learning_rate,lmbda,gamma,clip_param,value_param,entropy_act_param)

        self.memory = True

    def remember(self, state, action, hist_mem):
        action_one_hot = torch.zeros((1, 7)).to(device)
        action_one_hot[0, action] = 1
        obs = torch.FloatTensor(state).to(device)
        memory = self.model.remember(obs, action_one_hot, hist_mem)
        return memory


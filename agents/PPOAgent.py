#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions

from models.SharedCNN import SharedCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPOAgent():
    def __init__(self, state_dim, action_dim, question_rnn, vocab, hidden_dim=64, learning_rate=0.001,
                 gamma=0.99, clip_param=0.2, value_param=1, entropy_param=0.01,
                 lmbda=0.95, backward_epochs=3):
        self.vocab = vocab
        self.vocab_size = len(vocab)

        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.lmbda = lmbda

        self.model = SharedCNN(question_rnn, vocab_size=self.vocab_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.optimizer_qa = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.clip_param = clip_param
        self.entropy_param = entropy_param
        self.value_param = value_param
        self.T = 1

        self.done = True
        self.data = []
        self.backward_epochs = backward_epochs

    def ask(self, observation):
        # Will this be taking the history of questions?
        # And history of states?
        qa_history = torch.rand((1, 64))
        observation = torch.FloatTensor(observation).to(device)
        tkn_idxs, hx, log_probs_qa, entropy_qa = self.model(observation, None, None, qa_history, flag="question")
        output = str()
        for word in tkn_idxs:
            output += str(self.vocab[word])
        return output, hx, log_probs_qa, entropy_qa

    def act(self, observation, ans, hx):
        # Calculate policy
        observation = torch.FloatTensor(observation).to(device)
        ans = torch.FloatTensor(ans).view((-1, 2)).to(device)
        logits = self.model(observation, ans, hx, None, flag="policy")
        action_prob = F.softmax(logits.squeeze() / self.T, dim=-1)
        dist = distributions.Categorical(action_prob)
        action = dist.sample()
        probs = action_prob[action]  # Policy log prob
        entropy = dist.entropy()  # Entropy regularizer
        return action.detach().item(), probs, entropy

    def update(self):
        obs, ans, hx, a, reward, next_obs, done, log_prob, entropy = self.get_batch()

        for i in range(self.backward_epochs):
            # Get current V
            V_pred = self.model(obs, ans, hx, None, flag="value").squeeze()
            # Get next V
            next_V_pred = self.model(next_obs, ans, hx, None, flag="value").squeeze()

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
            logits = self.model(obs, ans, hx, None, flag="policy")
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
            total_loss.backward(retain_graph=True)
            self.optimizer.step()
        return total_loss.item()

    def update_QA(self, reward, log_prob, entropy):
        for i in range(1):
            # Reinforce Loss - TODO: check entropy parametr to avoid deterministic collapse
            total_loss = -(reward * torch.cat(log_prob).mean()
                           + 0.05 * entropy).to(device)
            # Update params
            self.optimizer.zero_grad()
            total_loss.backward(retain_graph=True)
            self.optimizer_qa.step()
        return total_loss.item()

    def store(self, transition):
        self.data.append(transition)

    def get_batch(self):
        obs_batch = []
        ans_batch = []
        hx_batch = []
        a_batch = []
        r_batch = []
        next_obs_batch = []
        prob_batch = []
        entropy_batch = []
        done_batch = []
        for transition in self.data:
            obs, ans, hx, action, reward, next_obs, prob, entropy, done = transition

            obs_batch.append(obs)
            ans_batch.append(ans)
            hx_batch.append(hx)
            a_batch.append([action])
            r_batch.append([reward])
            next_obs_batch.append(next_obs)
            prob_batch.append([prob])
            entropy_batch.append([entropy])
            done_bool = 0 if done else 1
            done_batch.append([done_bool])

        obs = torch.FloatTensor(obs_batch).to(device)
        ans = torch.FloatTensor(ans_batch).to(device)
        hx = torch.cat(hx_batch)
        a = torch.FloatTensor(a_batch).to(device)
        r = torch.FloatTensor(r_batch).to(device)
        next_obs = torch.FloatTensor(next_obs_batch).to(device)
        prob = torch.FloatTensor(prob_batch).to(device)
        entropy = torch.FloatTensor(entropy_batch).to(device)
        done = torch.FloatTensor(done_batch).to(device)

        self.data = []

        return obs, ans, hx, a, r, next_obs, done, prob, entropy

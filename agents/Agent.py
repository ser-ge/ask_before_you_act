#!/usr/bin/env python3

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions

from utils.Trainer import Transition


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent:
    def __init__(self, model, learning_rate=0.001, lmbda=0.95, gamma=0.99,
                 clip_param=0.2, value_param=1, entropy_act_param=0.01,
                 policy_qa_param=1, entropy_qa_param=0.05):

        self.gamma = gamma
        self.lmbda = lmbda

        self.clip_param = clip_param
        self.entropy_act_param = entropy_act_param
        self.value_param = value_param
        self.policy_qa_param = policy_qa_param
        self.entropy_qa_param = entropy_qa_param

        self.T = 1

        self.model = model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.optimizer_qa = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.done = True
        self.data = []

    def ask(self, observation, hidden_hist_mem):
        observation = torch.FloatTensor(observation).to(device)
        tokens, hidden_q, log_probs_qa, entropy_qa = self.model.gen_question(observation, hidden_hist_mem)
        output = ' '.join(tokens)
        return output, hidden_q, log_probs_qa, entropy_qa

    def act(self, observation, ans, hidden_q):
        # Calculate policy
        observation = torch.FloatTensor(observation).to(device)
        ans = torch.FloatTensor(ans).view((-1, 2)).to(device)
        logits = self.model.policy(observation, ans, hidden_q)
        action_prob = F.softmax(logits.squeeze() / self.T, dim=-1)
        dist = distributions.Categorical(action_prob)
        action = dist.sample()
        probs = action_prob[action]  # Policy log prob
        entropy = dist.entropy()  # Entropy regularizer
        return action.detach().item(), probs, entropy

    def update(self):
        state, answer, hidden_q, action, reward, reward_qa, next_state, \
        log_prob_act, log_prob_qa, entropy_act, entropy_qa, done, hidden_hist_mem, \
        cell_hist_mem, next_hidden_hist_mem, next_cell_hist_mem = self.get_batch()

        # Get current V
        V_pred = self.model.value(state, answer, hidden_q).squeeze()

        # Get next V
        next_V_pred = self.model.value(next_state, answer, hidden_q).squeeze()

        # Compute TD error
        target = reward.squeeze().to(device) + self.gamma * next_V_pred * done.squeeze().to(device)
        td_error = (target - V_pred).detach()

        # Generalised Advantage Estimation
        advantage = self.gae(td_error)

        # Clipped PPO Policy Loss
        L_clip = self.clip_loss(action, advantage, answer, log_prob_act, state, hidden_q)

        # Entropy regularizer
        L_entropy = self.entropy_act_param * entropy_act.detach().mean()

        # Value function loss
        L_value = self.value_param * F.smooth_l1_loss(V_pred, target.detach())

        # Q&A Loss
        L_policy_qa = self.policy_qa_param * (reward_qa + advantage.squeeze()) * log_prob_qa
        L_entropy_qa = self.entropy_qa_param * entropy_qa
        L_qa = (L_policy_qa + L_entropy_qa).mean().to(device)

        # Total loss
        total_loss = -(L_clip + L_qa - L_value + L_entropy).to(device)

        # Update params
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

    def clip_loss(self, action, advantage, answer, log_prob_act, state, hidden_q):
        logits = self.model.policy(state, answer, hidden_q)
        probs = F.softmax(logits, dim=-1)
        pi_a = probs.squeeze(1).gather(1, action.long())
        ratio = torch.exp(torch.log(pi_a) - torch.log(log_prob_act))
        surrogate1 = ratio * advantage
        surrogate2 = advantage * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
        L_clip = torch.min(surrogate1, surrogate2).mean()
        return L_clip

    def init_memory(self):
        return (torch.rand(1, self.model.mem_hidden_dim),
                torch.rand(1, self.model.mem_hidden_dim))

    def store(self, transition):
        self.data.append(transition)

    def get_batch(self):
        trans = Transition(*zip(*self.data))

        state = torch.FloatTensor(trans.state).to(device)
        answer = torch.FloatTensor(trans.answer).to(device)
        hidden_q = torch.cat(trans.hidden_q)
        action = torch.FloatTensor(trans.action).to(device).view(-1, 1)
        reward = torch.FloatTensor(trans.reward).to(device).view(-1, 1)
        reward_qa = torch.FloatTensor(trans.reward_qa).to(device)
        next_state = torch.FloatTensor(trans.next_state).to(device)
        log_prob_act = torch.FloatTensor(trans.log_prob_act).to(device).view(-1, 1)
        log_prob_qa = torch.stack(list(map(lambda t: torch.stack(t).mean().to(device), trans.log_prob_qa)))
        entropy_act = torch.FloatTensor(trans.entropy_act).to(device).view(-1, 1)
        entropy_qa = torch.FloatTensor(trans.entropy_qa).to(device)
        done = ~torch.BoolTensor(trans.done).to(device).view(-1, 1)  # You need the tilde!
        hidden_hist_mem = torch.cat(trans.hidden_hist_mem)
        cell_hist_mem = torch.cat(trans.cell_hist_mem)
        next_hidden_hist_mem = torch.cat(trans.next_hidden_hist_mem)
        next_cell_hist_mem = torch.cat(trans.next_cell_hist_mem)

        self.data = []

        return (state, answer, hidden_q, action, reward, reward_qa,
                next_state, log_prob_act, log_prob_qa, entropy_act, entropy_qa,
                done,  hidden_hist_mem, cell_hist_mem, next_hidden_hist_mem,
                next_cell_hist_mem)


class AgentMem(Agent):
    def __init__(self, model, learning_rate=0.001, lmbda=0.95, gamma=0.99,
                 clip_param=0.2, value_param=1, entropy_act_param=0.01,
                 policy_qa_param=1, entropy_qa_param=0.05):

        self.action_memory = False

        super().__init__(model, learning_rate, lmbda, gamma,
                 clip_param, value_param, entropy_act_param,
                 policy_qa_param, entropy_qa_param)

    def remember(self, state, answer, hidden_q, hist_mem):
        obs = torch.FloatTensor(state).to(device)
        answer = torch.FloatTensor(answer).view((-1, 2)).to(device)
        memory = self.model.remember(obs, answer, hidden_q, hist_mem)
        return memory
    
    
class AgentMemAction(Agent):
    def __init__(self, model, learning_rate=0.001, lmbda=0.95, gamma=0.99,
                 clip_param=0.2, value_param=1, entropy_act_param=0.01,
                 policy_qa_param=1, entropy_qa_param=0.05):

        self.action_memory = True

        super().__init__(model, learning_rate, lmbda, gamma,
                 clip_param, value_param, entropy_act_param,
                 policy_qa_param, entropy_qa_param)


    def remember(self, state, action, answer, hidden_q, hist_mem):
        action = torch.FloatTensor([action]).unsqueeze(0)
        obs = torch.FloatTensor(state).to(device)
        answer = torch.FloatTensor(answer).view((-1, 2)).to(device)
        memory = self.model.remember(obs, action, answer, hidden_q, hist_mem)
        return memory



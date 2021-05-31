#!/usr/bin/env python3

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions

from utils.Trainer import Transition


device = "cpu"


class Agent:
    def __init__(self, model, learning_rate=0.001, lmbda=0.95, gamma=0.99,
                 clip_param=0.2, value_param=1, entropy_act_param=0.01,
                 policy_qa_param=1, advantage_qa_param=0.5, entropy_qa_param=0.05):

        self.gamma = gamma
        self.lmbda = lmbda

        self.clip_param = clip_param
        self.entropy_act_param = entropy_act_param
        self.value_param = value_param
        self.policy_qa_param = policy_qa_param
        self.advantage_qa_param = advantage_qa_param
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

    def act(self, observation, ans, hidden_q, hidden_hist):
        # Calculate policy
        _ = hidden_hist # does nothing, just accepts
        observation = torch.FloatTensor(observation).to(device)
        ans = torch.FloatTensor(ans).view((-1, 2)).to(device)
        logits = self.model.policy(observation, ans, hidden_q)
        # does nothing, just accepts
        action_prob = F.softmax(logits.squeeze() / self.T, dim=-1)
        dist = distributions.Categorical(action_prob)
        action = dist.sample()
        probs = action_prob[action]  # Policy log prob
        entropy = dist.entropy()  # Entropy regularizer
        return action.detach().item(), probs, entropy

    def update(self):

        current_trans, next_trans = self.get_batch()

        state, answer, hidden_q, action, reward, reward_qa, \
        log_prob_act, log_prob_qa, entropy_act, entropy_qa, \
        done, _, hidden_hist_mem, cell_hist_mem = current_trans

        next_state, next_answer, next_hidden_q, *_ = next_trans

        # Get current V
        V_pred = self.model.value(state, answer, hidden_q).squeeze()

        # Get next V
        next_V_pred = self.model.value(next_state, next_answer, next_hidden_q).squeeze()

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

        discounted_reward = torch.Tensor([(self.gamma**i) * reward.squeeze()[-1] for i in range(reward.shape[0])])

        R_t = torch.cumsum(discounted_reward,0).T


        L_policy_qa = ((self.policy_qa_param * reward_qa +
                        self.advantage_qa_param * R_t) * log_prob_qa).mean()


        L_entropy_qa = self.entropy_qa_param * entropy_qa.mean()
        L_qa = (L_policy_qa + L_entropy_qa).to(device)

        # Total loss
        total_loss = -(L_clip + L_qa - L_value + L_entropy).to(device)

        # Update paramss
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item(), (L_clip, L_value, L_entropy, L_policy_qa, L_entropy_qa)

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

        current_trans = Transition(*zip(*self.data))
        current_trans = self.transition_to_tensors(current_trans)

        next_data = self.data[1:]
        next_trans = Transition(*zip(*next_data))
        next_trans = self.transition_to_tensors(next_trans)
        next_trans = Transition(*list(map(expand_zeros, next_trans)))

        self.data = []
        return current_trans, next_trans


    def transition_to_tensors(self, trans):
        state = torch.FloatTensor(trans.state).to(device)
        answer = torch.FloatTensor(trans.answer).to(device)
        hidden_q = torch.cat(trans.hidden_q)
        action = torch.FloatTensor(trans.action).to(device).view(-1, 1)
        reward = torch.FloatTensor(trans.reward).to(device).view(-1, 1)
        reward_qa = torch.FloatTensor(trans.reward_qa).to(device)
        # next_state = torch.FloatTensor(trans.next_state).to(device)
        log_prob_act = torch.FloatTensor(trans.log_prob_act).to(device).view(-1, 1)
        log_prob_qa = torch.stack(list(map(lambda t: torch.stack(t).mean().to(device), trans.log_prob_qa)))
        entropy_act = torch.FloatTensor(trans.entropy_act).to(device).view(-1, 1)
        entropy_qa = torch.FloatTensor(trans.entropy_qa).to(device)
        done = ~torch.BoolTensor(trans.done).to(device).view(-1, 1)  # You need the tilde!
        hidden_hist_mem = torch.cat(trans.hidden_hist_mem)
        cell_hist_mem = torch.cat(trans.cell_hist_mem)
        q_embedding = torch.stack(trans.q_embedding)

        return Transition(state, answer, hidden_q, action, reward, reward_qa,
                log_prob_act, log_prob_qa, entropy_act, entropy_qa,
                done, q_embedding, hidden_hist_mem, cell_hist_mem)

class AgentMem(Agent):
    def __init__(self, model, learning_rate=0.001, lmbda=0.95, gamma=0.99,
                 clip_param=0.2, value_param=1, entropy_act_param=0.01,
                 policy_qa_param=1, advantage_qa_param=0.5, entropy_qa_param=0.05):
        super().__init__(model, learning_rate, lmbda, gamma,
                 clip_param, value_param, entropy_act_param,
                 policy_qa_param, advantage_qa_param, entropy_qa_param)

    def remember(self, state, action, answer, hidden_q, hist_mem):
        action_one_hot = torch.zeros((1, 7)).to(device)
        action_one_hot[0, action] = 1
        obs = torch.FloatTensor(state).to(device)
        answer = torch.FloatTensor(answer).view((-1, 2)).to(device)
        memory = self.model.remember(obs, action_one_hot, answer, hidden_q, hist_mem)
        return memory

    def act(self, observation, ans, hidden_q, hidden_hist_mem):
        # Calculate policy

        # note, act doesn't actually USE hidden_hist_mem here
        # it's just here to make trainer look nicer
        # it gets passed to the policy, but it will similarly be ignored there too

        observation = torch.FloatTensor(observation).to(device)
        ans = torch.FloatTensor(ans).view((-1, 2)).to(device)
        logits = self.model.policy(observation, ans, hidden_q, hidden_hist_mem)
        action_prob = F.softmax(logits.squeeze() / self.T, dim=-1)
        dist = distributions.Categorical(action_prob)
        action = dist.sample()
        probs = action_prob[action]  # Policy log prob
        entropy = dist.entropy()  # Entropy regularizer
        return action.detach().item(), probs, entropy

    def clip_loss(self, action, advantage, answer, log_prob_act, state, hidden_q):
        # hidden_hist_mem will be a placeholder here, passed to the policy,
        # but then subsequently ignored by the policy, if we have initialised it to YES use
        # memory, but not explicitly into the policy...

        hidden_hist_mem = torch.zeros(128)

        logits = self.model.policy(state, answer, hidden_q, hidden_hist_mem)
        probs = F.softmax(logits, dim=-1)
        pi_a = probs.squeeze(1).gather(1, action.long())
        ratio = torch.exp(torch.log(pi_a) - torch.log(log_prob_act))
        surrogate1 = ratio * advantage
        surrogate2 = advantage * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
        L_clip = torch.min(surrogate1, surrogate2).mean()
        return L_clip

class AgentExpMem(Agent):
    def __init__(self, model, learning_rate=0.001, lmbda=0.95, gamma=0.99,
                 clip_param=0.2, value_param=1, entropy_act_param=0.01,
                 policy_qa_param=1, advantage_qa_param=0.5, entropy_qa_param=0.05):
        super().__init__(model, learning_rate, lmbda, gamma,
                 clip_param, value_param, entropy_act_param,
                 policy_qa_param, advantage_qa_param, entropy_qa_param)

        self.action_memory = True

    def act(self, observation, ans, hidden_q, hidden_hist_mem):
        # Calculate policy
        observation = torch.FloatTensor(observation).to(device)
        ans = torch.FloatTensor(ans).view((-1, 2)).to(device)
        logits = self.model.policy(observation, ans, hidden_q, hidden_hist_mem)
        action_prob = F.softmax(logits.squeeze() / self.T, dim=-1)
        dist = distributions.Categorical(action_prob)
        action = dist.sample()
        probs = action_prob[action]  # Policy log prob
        entropy = dist.entropy()  # Entropy regularizer
        return action.detach().item(), probs, entropy

    def update(self):
        current_trans, next_trans = self.get_batch()

        state, answer, hidden_q, action, reward, reward_qa, \
        log_prob_act, log_prob_qa, entropy_act, entropy_qa, done, _, hidden_hist_mem, cell_hist_mem  = current_trans

        next_state, next_answer, next_hidden_q, *_ = next_trans
        *_ , next_hidden_hist_mem, cell_hist_mem = next_trans

        # Get next V
        # Get current V
        V_pred = self.model.value(state, answer, hidden_q, hidden_hist_mem).squeeze()

        # Get next V
        next_V_pred = self.model.value(next_state, next_answer, next_hidden_q, next_hidden_hist_mem).squeeze()

        # Compute TD error
        target = reward.squeeze().to(device) + self.gamma * next_V_pred * done.squeeze().to(device)
        td_error = (target - V_pred).detach()

        # Generalised Advantage Estimation
        advantage = self.gae(td_error)

        # Clipped PPO Policy Loss
        # TODO - try to unify clip_loss wit and w/o hidden_hist_mem
        L_clip = self.clip_loss(action, advantage, answer, log_prob_act, state, hidden_q,hidden_hist_mem)

        # Entropy regularizer
        L_entropy = self.entropy_act_param * entropy_act.detach().mean()

        # Value function loss
        L_value = self.value_param * F.smooth_l1_loss(V_pred, target.detach())

        # Q&A Loss


        R_t = torch.cumsum(discounted_reward,0).T


        L_policy_qa = ((self.policy_qa_param * reward_qa +
                        self.advantage_qa_param * R_t) * log_prob_qa).mean()


        L_entropy_qa = self.entropy_qa_param * entropy_qa.mean()
        L_qa = (L_policy_qa + L_entropy_qa).to(device)

        # Total loss
        total_loss = -(L_clip + L_qa - L_value + L_entropy).to(device)

        # Update paramss
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item(), (L_clip, L_value, L_entropy, L_policy_qa, L_entropy_qa)

    def clip_loss(self, action, advantage, answer, log_prob_act, state, hidden_q, hidden_hist):
        logits = self.model.policy(state, answer, hidden_q,hidden_hist)
        # TODO - try to unify clip_loss wit and w/o hidden_hist_mem
        probs = F.softmax(logits, dim=-1)
        pi_a = probs.squeeze(1).gather(1, action.long())
        ratio = torch.exp(torch.log(pi_a) - torch.log(log_prob_act))
        surrogate1 = ratio * advantage
        surrogate2 = advantage * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
        L_clip = torch.min(surrogate1, surrogate2).mean()
        return L_clip

    def remember(self, state, action, answer, hidden_q, hist_mem):
        action_one_hot = torch.zeros((1, 7)).to(device)
        action_one_hot[0, action] = 1
        obs = torch.FloatTensor(state).to(device)
        answer = torch.FloatTensor(answer).view((-1, 2)).to(device)
        memory = self.model.remember(obs, action_one_hot, answer, hidden_q, hist_mem)
        return memory




class AgentExpMemEmbed(Agent):
    def __init__(self, model, learning_rate=0.001, lmbda=0.95, gamma=0.99,
                 clip_param=0.2, value_param=1, entropy_act_param=0.01,
                 policy_qa_param=1, advantage_qa_param=0.5, entropy_qa_param=0.05):
        super().__init__(model, learning_rate, lmbda, gamma,
                 clip_param, value_param, entropy_act_param,
                 policy_qa_param, advantage_qa_param, entropy_qa_param)

        self.action_memory = True

    def act(self, observation, ans, hidden_q, hidden_hist_mem, q_embedding):
        # Calculate policy
        observation = torch.FloatTensor(observation).to(device)
        ans = torch.FloatTensor(ans).view((-1, 2)).to(device)
        q_embedding = q_embedding.unsqueeze(0)
        logits = self.model.policy(observation, ans, hidden_q, hidden_hist_mem, q_embedding)
        action_prob = F.softmax(logits.squeeze() / self.T, dim=-1)
        dist = distributions.Categorical(action_prob)
        action = dist.sample()
        probs = action_prob[action]  # Policy log prob
        entropy = dist.entropy()  # Entropy regularizer
        return action.detach().item(), probs, entropy

    def update(self):
        current_trans, next_trans = self.get_batch()

        state, answer, hidden_q, action, reward, reward_qa, \
        log_prob_act, log_prob_qa, entropy_act, entropy_qa, done, q_embedding, hidden_hist_mem, cell_hist_mem = current_trans

        next_state, next_answer, next_hidden_q, *_ = next_trans
        *_ , next_q_embedding, next_hidden_hist_mem, cell_hist_mem,  = next_trans

        # Get next V
        # Get current V
        V_pred = self.model.value(state, answer, hidden_q, hidden_hist_mem, q_embedding).squeeze()

        # Get next V
        next_V_pred = self.model.value(next_state, next_answer, next_hidden_q, next_hidden_hist_mem, next_q_embedding).squeeze()

        # Compute TD error
        target = reward.squeeze().to(device) + self.gamma * next_V_pred * done.squeeze().to(device)
        td_error = (target - V_pred).detach()

        # Generalised Advantage Estimation
        advantage = self.gae(td_error)

        # Clipped PPO Policy Loss
        # TODO - try to unify clip_loss wit and w/o hidden_hist_mem
        L_clip = self.clip_loss(action, advantage, answer, log_prob_act, state, hidden_q,hidden_hist_mem, q_embedding)

        # Entropy regularizer
        L_entropy = self.entropy_act_param * entropy_act.detach().mean()

        # Value function loss
        L_value = self.value_param * F.smooth_l1_loss(V_pred, target.detach())

        # Q&A Loss

        discounted_reward = torch.Tensor([(self.gamma**i) * reward.squeeze()[-1] for i in range(reward.shape[0])])

        R_t = torch.cumsum(discounted_reward,0).T


        L_policy_qa = ((self.policy_qa_param * reward_qa +
                        self.advantage_qa_param * R_t) * log_prob_qa).mean()


        L_entropy_qa = self.entropy_qa_param * entropy_qa.mean()
        L_qa = (L_policy_qa + L_entropy_qa).to(device)

        # Total loss
        total_loss = -(L_clip + L_qa - L_value + L_entropy).to(device)

        # Update paramss
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item(), (L_clip, L_value, L_entropy, L_policy_qa, L_entropy_qa)

    def clip_loss(self, action, advantage, answer, log_prob_act, state, hidden_q, hidden_hist, q_embedding):
        logits = self.model.policy(state, answer, hidden_q,hidden_hist, q_embedding)
        # TODO - try to unify clip_loss wit and w/o hidden_hist_mem
        probs = F.softmax(logits, dim=-1)
        pi_a = probs.squeeze(1).gather(1, action.long())
        ratio = torch.exp(torch.log(pi_a) - torch.log(log_prob_act))
        surrogate1 = ratio * advantage
        surrogate2 = advantage * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
        L_clip = torch.min(surrogate1, surrogate2).mean()
        return L_clip

    def remember(self, state, action, answer, hidden_q, hist_mem):
        action_one_hot = torch.zeros((1, 7)).to(device)
        action_one_hot[0, action] = 1
        obs = torch.FloatTensor(state).to(device)
        answer = torch.FloatTensor(answer).view((-1, 2)).to(device)
        memory = self.model.remember(obs, action_one_hot, answer, hidden_q, hist_mem)
        return memory

    def ask(self, observation, hidden_hist_mem):
        observation = torch.FloatTensor(observation).to(device)
        tokens, hidden_q, log_probs_qa, entropy_qa, q_embedding = self.model.gen_question(observation, hidden_hist_mem)
        output = ' '.join(tokens)
        return output, hidden_q, log_probs_qa, entropy_qa, q_embedding


def expand_zeros(tensor):
    pad = torch.zeros_like(tensor[0]).unsqueeze(0)
    return torch.cat((tensor, pad), 0)



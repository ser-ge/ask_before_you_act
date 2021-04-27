#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.distributions as distributions
from language_model.model import Model as QuestionRNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BrainNet(nn.Module):
    def __init__(self, question_rnn, action_dim=7):
        super().__init__()

        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU())

        self.mem_hidden_dim = 64

        self.policy_head = nn.Linear(194, action_dim)  # 194 is 128 of hx + 2 of answer + 64 obs CNN
        self.value_head = nn.Linear(194, 1)  # 194 is 128 of hx + 2 of answer + 64 obs CNN

        # CNN output is 64 dims
        # Assuming qa_history is also 64
        self.question_rnn = question_rnn
        self.softmax = nn.Softmax(dim=-1)

    def policy(self, obs, answer, hidden_q):
        """
        hidden_q : last hidden state
        """
        encoded_obs = self.encode_obs(obs)
        x = torch.cat((encoded_obs, answer, hidden_q), 1)
        action_policy = self.policy_head(x)
        return action_policy

    def value(self, obs, answer, hidden_q):
        encoded_obs = self.encode_obs(obs)
        x = torch.cat((encoded_obs, answer, hidden_q), 1)
        state_value = self.value_head(x)
        return state_value

    def gen_question(self, obs, hidden_hist_mem):
        encoded_obs = self.encode_obs(obs)
        hx = torch.cat((encoded_obs, hidden_hist_mem.view(-1, 64)), 1)
        cx = torch.randn(hx.shape).to(device)  # (batch, hidden_size)
        entropy_qa = 0
        log_probs_qa = []
        words = ['<sos>']

        memory = (hx, cx)

        while words[-1] != '<eos>':
            x = torch.tensor(self.question_rnn.dataset.word_to_index[words[-1]]).unsqueeze(0).to(device)
            logits, memory = self.question_rnn.process_single_input(x, memory)
            dist = self.softmax(logits.squeeze())
            m = distributions.Categorical(dist)
            tkn_idx = m.sample()
            log_probs_qa.append(m.log_prob(tkn_idx))
            entropy_qa += m.entropy().item()
            word = self.question_rnn.dataset.index_to_word[tkn_idx.item()]
            words.append(word)

        entropy_qa /= len(words)

        last_hidden_state = memory[0]
        output = words[1:-1]  # remove sos and eos

        return output, last_hidden_state, log_probs_qa, entropy_qa

    def encode_obs(self, obs):
        x = obs.view(-1, 3, 7, 7)  # x: (batch, C_in, H_in, W_in)
        obs_encoding = self.image_conv(x).view(-1, 64)  # x: (batch, hidden)
        return obs_encoding


class BrainNetMem(BrainNet):
    def __init__(self, question_rnn):
        super().__init__(question_rnn, action_dim=7)
        # here only 194 because we don't include action
        self.memory_rnn = nn.LSTMCell(194, self.mem_hidden_dim)

    def remember(self, obs, answer, hidden_q, memory):
        encoded_obs = self.encode_obs(obs)
        x = torch.cat((encoded_obs, answer, hidden_q), 1)
        return self.memory_rnn(x, memory)

class BrainNetMemAction(BrainNet):
    def __init__(self, question_rnn):
        super().__init__(question_rnn, action_dim=7)
        # here 195 because we do include action
        self.memory_rnn = nn.LSTMCell(195, self.mem_hidden_dim)

    def remember(self, obs, action, answer, hidden_q, memory):
        encoded_obs = self.encode_obs(obs)
        x = torch.cat((encoded_obs, action, answer, hidden_q), 1)
        return self.memory_rnn(x, memory)

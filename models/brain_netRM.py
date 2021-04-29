#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.distributions as distributions

from language_model.model import Model as QuestionRNN

device = "cpu"

class BrainNet(nn.Module):
    def __init__(self, question_rnn, action_dim=7, hidden_q_dim=128, mem_hidden_dim=64):
        super().__init__()

        # what size do you want the output of your CNN to be?
        # the CNN observes the environment, and converts the 7*7*3 tensor to a 1,64 encoding
        self.cnn_encoding_dim = 64

        # the dimension of the encoding of the qa pairs (which also takes some state,action memory into account..)
        self.hidden_q_dim = hidden_q_dim

        # the dimension of the encoding of the memory of the history
        self.mem_hidden_dim = mem_hidden_dim

        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, self.cnn_encoding_dim, (2, 2)),
            nn.ReLU())

        # the policy takes the encoding of the obs, plus the answer you got back, plus the
        # encoding of the qa pairs history
        # note that HERE, the policy does NOT take the explicit memory
        # however, this is added later on (inherited classes below).
        self.policy_input_dim = self.cnn_encoding_dim + 2 + self.hidden_q_dim

        self.policy_head = nn.Linear(self.policy_input_dim, action_dim)  # 194 is 128 of hx + 2 of answer + 64 obs CNN
        self.value_head = nn.Linear(self.policy_input_dim, 1)  # 194 is 128 of hx + 2 of answer + 64 obs CNN

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

    def gen_question(self, obs, encoded_memory):
        '''
        generate a question to ask the oracle
        note that this method involves the question_rnn
        and note that the question_rnn takes as an input a history of your
        observations and actions
        so, this ALREADY in a sense gives the agent a concept of memory
        '''
        encoded_obs = self.encode_obs(obs)
        hx = torch.cat((encoded_obs, encoded_memory.view(-1, self.mem_hidden_dim)), 1)
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
            if len(words) > 6: break

        entropy_qa /= len(words)

        last_hidden_state = memory[0]
        output = words[1:-1]  # remove sos and eos

        return output, last_hidden_state, log_probs_qa, entropy_qa

    def encode_obs(self, obs):
        x = obs.view(-1, 3, 7, 7)  # x: (batch, C_in, H_in, W_in)
        obs_encoding = self.image_conv(x).view(-1, self.cnn_encoding_dim)  # x: (batch, hidden)
        return obs_encoding


class BrainNetMem(BrainNet):
    def __init__(self, question_rnn, action_dim=7):
        super().__init__(question_rnn, action_dim)
        self.memory_rnn = nn.LSTMCell(self.cnn_encoding_dim  + action_dim + 2 + self.hidden_q_dim,
                                      self.mem_hidden_dim)

    def remember(self, obs, action, answer, hidden_q, memory):
        encoded_obs = self.encode_obs(obs)
        x = torch.cat((encoded_obs, action, answer, hidden_q), 1)
        return self.memory_rnn(x, memory)

    def policy(self, obs, answer, hidden_q, hidden_hist_mem):
        """
        hidden_q : last hidden state
        """
        _ = hidden_hist_mem # not doing anything with this
        # just taking it in here to make Trainer read cleaner
        encoded_obs = self.encode_obs(obs)
        x = torch.cat((encoded_obs, answer, hidden_q), 1)
        action_policy = self.policy_head(x)
        return action_policy


class BrainNetExpMem(BrainNetMem):
    def __init__(self, question_rnn, action_dim=7):
        super().__init__(question_rnn, action_dim)
        # self.mem_hidden_dim is the explicit memory connection
        # 265 is 64 obs CNN + 7 one hot action + 2 of answer + 128 of Q&A hx + 64 from explicit memory
        self.policy_input_dim = self.cnn_encoding_dim  + 2 + self.hidden_q_dim \
                                + self.mem_hidden_dim

        self.policy_head = nn.Linear(self.policy_input_dim, action_dim)
        self.value_head = nn.Linear(self.policy_input_dim, 1)

    def policy(self, obs, answer, hidden_q, hidden_hist_mem):
        """
        hidden_q : last hidden state§
        """
        encoded_obs = self.encode_obs(obs)
        x = torch.cat((encoded_obs, answer, hidden_q, hidden_hist_mem), 1)
        action_policy = self.policy_head(x)
        return action_policy

    def value(self, obs, answer, hidden_q, hidden_hist_mem):
        encoded_obs = self.encode_obs(obs)
        x = torch.cat((encoded_obs, answer, hidden_q,hidden_hist_mem), 1)
        state_value = self.value_head(x)
        return state_value

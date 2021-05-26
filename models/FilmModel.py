#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.distributions as distributions

from language_model.model import Model as QuestionRNN

device = "cpu"

class FilmNet(nn.Module):
    def __init__(self, question_rnn, action_dim=7, hidden_q_dim=128, mem_hidden_dim=64, n_res_blocks=2, image_conv_dim=16):
        super().__init__()

        # what size do you want the output of your CNN to be?
        # the CNN observes the environment, and converts the 7*7*3 tensor to a 1,64 encoding
        self.cnn_encoding_dim = 64

        self.n_res_blocks = n_res_blocks

        self.hidden_q_dim = hidden_q_dim
        self.mem_hidden_dim = mem_hidden_dim

        self.memory_rnn = nn.LSTMCell(201,
                                      self.mem_hidden_dim)

        self.image_conv_dim = image_conv_dim
        # the dimension of the encoding of the qa pairs (which also takes some state,action memory into account..)

        # the dimension of the encoding of the memory of the history
        self.mem_hidden_dim = mem_hidden_dim

        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, image_conv_dim, (2, 2)))

        self.res_blocks = nn.ModuleList()

        for _ in range(n_res_blocks):
            self.res_blocks.append(ResBlock(image_conv_dim, image_conv_dim))


        self.embded_answer = nn.Sequential(
                nn.Linear(2, hidden_q_dim),
                nn.ReLU())

        self.film = nn.Linear(256, self.image_conv_dim *2 * n_res_blocks)

        # the policy takes the encoding of the obs, plus the answer you got back, plus the
        # encoding of the qa pairs history
        # note that HERE, the policy does NOT take the explicit memory
        # however, this is added later on (inherited classes below).
        self.policy_input_dim = image_conv_dim * 8

        self.policy_head = nn.Linear(self.policy_input_dim, action_dim)  # 194 is 128 of hx + 2 of answer + 64 obs CNN
        self.value_head = nn.Linear(self.policy_input_dim, 1)  # 194 is 128 of hx + 2 of answer + 64 obs CNN

        # CNN output is 64 dims
        # Assuming qa_history is also 64
        self.question_rnn = question_rnn
        self.softmax = nn.Softmax(dim=-1)



    def gen_question(self, obs, encoded_memory):
        '''
        generate a question to ask the oracle
        note that this method involves the question_rnn
        and note that the question_rnn takes as an input a history of your
        observations and actions
        so, this ALREADY in a sense gives the agent a concept of memory
        '''
        encoded_obs = self.encode_obs(obs).view(-1, self.image_conv_dim * 4)
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
        obs_encoding = self.image_conv(x)
        return obs_encoding

    def film_net(self, encoded_obs, hidden_q, answer):
        answer = self.embded_answer(answer)

        qa = torch.cat((hidden_q, answer),1)

        film = self.film(qa).chunk(self.n_res_blocks * 2, 1)
        for i, res_block in enumerate(self.res_blocks):
            encoded_obs = res_block(encoded_obs, film[i *2], film[i * 2 + 1])

        return encoded_obs

    def policy(self, obs, answer, hidden_q, hidden_hist_mem):
        """
        hidden_q : last hidden state§

        """
        encoded_obs = self.encode_obs(obs)
        conditioned_state = self.film_net(encoded_obs, hidden_q, answer)
        conditioned_state = conditioned_state.view(-1, self.image_conv_dim*4)

        x = torch.cat((conditioned_state, hidden_hist_mem), 1)

        action_policy = self.policy_head(x)
        return action_policy


    def value(self, obs, answer, hidden_q, hidden_hist_mem):

        encoded_obs = self.encode_obs(obs)
        conditioned_state = self.film_net(encoded_obs, hidden_q, answer)
        conditioned_state = conditioned_state.view(-1, self.image_conv_dim*4)

        x = torch.cat((conditioned_state, hidden_hist_mem), 1)

        state_value = self.value_head(x)
        return state_value

    def remember(self, obs, action, answer, hidden_q, memory):

        encoded_obs = self.encode_obs(obs)
        conditioned_state = self.film_net(encoded_obs, hidden_q, answer)
        conditioned_state = conditioned_state.view(-1, self.image_conv_dim *4)

        x = torch.cat((conditioned_state, action, answer, hidden_q), 1)
        return self.memory_rnn(x, memory)






class FiLMBlock(nn.Module):
    """
    apply gamma and beta to activations x
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, gamma, beta):
        beta = beta.view(x.size(0), x.size(1), 1, 1)
        gamma = gamma.view(x.size(0), x.size(1), 1, 1)
        x = gamma * x + beta
        return x

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, (1,1))
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, (1,1))
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.film = FiLMBlock()
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, beta, gamma):
        x = self.conv1(x)
        x = self.relu1(x)
        identity = x

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.film(x, beta, gamma)
        x = self.relu2(x)

        x = x + identity

        return x







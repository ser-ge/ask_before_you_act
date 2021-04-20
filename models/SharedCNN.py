#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.distributions as distributions
from language_model.model import Model as QuestionRNN

QUESTION_SAMPLING_TEMP = 0.9
class SharedCNN(nn.Module):
    def __init__(self, question_rnn, action_dim=7, vocab_size=10):
        super(SharedCNN, self).__init__()

        self.vocab_size = vocab_size

        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU())

        self.policy_head = nn.Linear(194, action_dim)  # 194 is 128 of hx + 2 of answer + 64 obs CNN
        self.value_head = nn.Linear(194, 1)  # 194 is 128 of hx + 2 of answer + 64 obs CNN

        # CNN output is 64 dims
        # Assuming qa_history is also 64
        self.question_rnn = question_rnn
        # self.question_rnn = nn.LSTMCell(self.vocab_size, 128)  # (input_size, hidden_size)
        # self.question_head = nn.Linear(128, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, answer, hx, qa_history, flag="policy"):
        # Shared Body
        x = x.view(-1, 3, 7, 7)  # x: (batch, C_in, H_in, W_in)
        batch = x.shape[0]
        x = self.image_conv(x).view(-1, 64)  # x: (batch, hidden)

        # Split heads
        if flag == "policy":
            x = torch.cat((x, answer, hx), 1)
            x_pol = self.policy_head(x)
            return x_pol
        elif flag == "value":
            x = torch.cat((x, answer, hx), 1)
            x_val = self.value_head(x)
            return x_val
        elif flag == "question":
            hx = torch.cat((x, qa_history.view(-1, 64)), 1)
            cx = torch.randn(hx.shape)  # (batch, hidden_size)
            output = []
            entropy_qa = 0
            log_probs_qa = []
            words = ['<sos>']

            memory = (hx, cx)

            for count in range(5):
                x = torch.tensor(self.question_rnn.dataset.word_to_index[words[-1]]).unsqueeze(0)
                logits, memory = self.question_rnn.process_single_input(x, memory)
                # p = torch.nn.functional.softmax(logits.squeeze(), dim=0)
                dist = self.softmax(logits)
                m = distributions.Categorical(dist)

                # tkn_idx = self.question_rnn.temperature_sampling(dist.detach(), QUESTION_SAMPLING_TEMP)
                tkn_idx = m.sample()

                log_probs_qa.append(m.log_prob(tkn_idx))
                entropy_qa += m.entropy().item()
                if count == 4:  # We want the last hidden state to include the 4th word
                    break

                words.append(self.question_rnn.dataset.index_to_word[tkn_idx])
                output.append(tkn_idx)

                # hx_tmp, cx_tmp = self.question_rnn(x, (hx, cx))
                # dist = self.softmax(self.question_head(hx_tmp))
                # m = distributions.Categorical(dist)
                # tkn_idx = m.sample()
                # log_probs_qa.append(m.log_prob(tkn_idx))
                # entropy_qa += m.entropy().item()
                # if count == 4:  # We want the last hidden state to include the 4th word
                #     break
                # output.append(tkn_idx)

            last_hidden_state = memory[0]
            output = torch.stack(output, dim=0)
            return output, last_hidden_state, log_probs_qa, entropy_qa

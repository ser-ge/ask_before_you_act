from collections import Callable

import torch
from torch import nn
from einops import rearrange
import numpy as np
from dataclasses import dataclass

@dataclass
class Config:
    epochs: int = 30
    batch_size : int = 256
    sequence_len: int = 10
    lstm_size : int = 128
    word_embed_dims : int = 128
    drop_out_prob : float = 0

cfg = Config()

class Model(nn.Module):
    def __init__(self, dataset, cfg=cfg):
        super(Model, self).__init__()
        self.lstm_size = cfg.lstm_size
        self.embedding_dim = cfg.word_embed_dims
        self.sequence_len = cfg.sequence_len

        self.dataset = dataset

        self.dropout = nn.Dropout(cfg.drop_out_prob)

        n_vocab = len(dataset.uniq_words)
        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=self.embedding_dim,
        )

        self.n_vocab = n_vocab
        self.lstm = nn.LSTMCell(self.embedding_dim, self.lstm_size)
        self.fc = nn.Linear(self.lstm_size, n_vocab)


    def process_single_input(self, word, memory):
        x_train = self.embedding(word)
        memory = self.lstm(x_train, memory)
        h, c = memory
        return self.fc(self.dropout(h)), memory



    def forward(self, sequences, memory):
        batch_size = sequences.shape[0]
        sequence_len = sequences.shape[1]
        output_seq = torch.empty((sequence_len,
                                  batch_size,
                                  self.n_vocab))

        for t in range(sequence_len):
            output_seq[t], memory = self.process_single_input(sequences[:,t], memory)


        output_seq = rearrange(output_seq, "seq_len batch_size v_size -> (batch_size seq_len) v_size" )

        return output_seq, memory

    def init_state(self, batch_size):
        return (torch.rand(batch_size, self.lstm_size),
                torch.rand(batch_size, self.lstm_size))

    def sample(self, temp=1):

        words = ['<sos>']
        self.eval()

        memory = self.init_state(1)

        while words[-1] != '<eos>':
            x = torch.tensor(self.dataset.word_to_index[words[-1]]).unsqueeze(0)
            logits, memory = self.process_single_input(x, memory)
            p = torch.nn.functional.softmax(logits.squeeze(), dim=0).detach().numpy()
            word_index = self.temperature_sampling(p, temp)
            words.append(self.dataset.index_to_word[word_index])

        return " ".join(words)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def temperature_sampling(self, conditional_probability, temperature=1.0):
        conditional_probability = np.asarray(conditional_probability).astype("float64")
        conditional_probability = np.log(conditional_probability) / temperature
        exp_preds = np.exp(conditional_probability)
        conditional_probability = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, conditional_probability, 1)
        return np.argmax(probas)

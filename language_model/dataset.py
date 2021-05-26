import torch
from collections import Counter
import json


class Dataset(torch.utils.data.Dataset):
    def __init__(
            self,
            cfg,
            path='./language_model/phrases.json'
    ):
        self.cfg = cfg
        self.path = path
        self.words = self.load_words()
        self.uniq_words = self.get_uniq_words()

        self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)}
        self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}

        self.words_indexes = [self.word_to_index[w] for w in self.words]

    def load_words(self):
        premises = list(json.load(open(self.path)))
        text = ["<sos> " + p + " <eos>" for p in premises]
        text = " ".join(text)
        return text.split(' ')

    def get_uniq_words(self):
        word_counts = Counter(self.words)
        return sorted(word_counts, key=word_counts.get, reverse=True)

    def __len__(self):
        return len(self.words_indexes) - self.cfg.sequence_len

    def __getitem__(self, index):
        return (
            torch.tensor(self.words_indexes[index:index + self.cfg.sequence_len]),
            torch.tensor(self.words_indexes[index + 1:index + self.cfg.sequence_len + 1]),
        )

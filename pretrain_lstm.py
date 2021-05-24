import torch

from language_model import train, Dataset, Model
from oracle.generator import gen_phrases
from typing import Callable
from dataclasses import dataclass

@dataclass
class Config:
    epochs: int = 50
    batch_size : int = 256
    sequence_len: int = 10
    lstm_size : int = 128
    word_embed_dims : int = 128
    drop_out_prob : float = 0
    gen_phrases : Callable = gen_phrases

cfg = Config()
dataset = Dataset(cfg)
model = Model(dataset, cfg)
train(dataset, model, cfg)


for temp in [0.5, 0.7, 0.9]:
    print("----")
    print("Predicting with temp: ", temp)
    print("----")
    for i in range(10):
        print(model.sample(temp))

model.save("./language_model/pre-trained.pth")



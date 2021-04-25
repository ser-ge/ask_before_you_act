import argparse
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from .model import Model
from .dataset import Dataset

def train(dataset, model, cfg):
    model.train()

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(cfg.epochs):
        state_h, state_c = model.init_state(cfg.batch_size)

        for batch, (x, y) in enumerate(dataloader):
            if y.shape[0] < cfg.batch_size: continue

            optimizer.zero_grad()

            y_pred, (state_h, state_c) = model(x, (state_h, state_c))
            loss = criterion(y_pred, y.view(-1))

            state_h = state_h.detach()
            state_c = state_c.detach()

            loss.backward()
            optimizer.step()

            print({ 'epoch': epoch, 'batch': batch, 'loss': loss.item() })

    return model



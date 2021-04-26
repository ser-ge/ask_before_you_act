# Environment
import random

import gym
import gym_minigrid
import torch
import numpy as np

from agents.Agent import Agent
from models.brain_net import BrainNet
from oracle.oracle import OracleWrapper
from utils.Trainer import train
from language_model import Dataset, Model as QuestionRNN
from oracle.generator import gen_phrases
from typing import Callable
from dataclasses import dataclass, asdict

import wandb

USE_WANDB = False


@dataclass
class Config:
    epochs: int = 30
    batch_size: int = 256
    sequence_len: int = 10
    lstm_size: int = 128
    word_embed_dims: int = 128
    drop_out_prob: float = 0
    gen_phrases: Callable = gen_phrases
    hidden_dim: float = 32
    lr: float = 0.001
    gamma: float = 0.99
    lmbda: float = 0.95
    clip: float = 0.1
    entropy_act_param: float = 0.1
    value_param: float = 1
    policy_qa_param: float = 1
    entropy_qa_param: float = 0.05
    N_eps: float = 2000
    train_log_interval: float = 25
    runs: float = 1
    env_name: str = "MiniGrid-Empty-5x5-v0"
    ans_random: bool = False
    undefined_error_reward: float = -0.1
    syntax_error_reward: float = -0.2
    pre_trained_lstm: bool = True


cfg = Config()

# %%

if USE_WANDB:
    wandb.init(project='ask_before_you_act', config=asdict(cfg))
    logger = wandb
else:
    class Logger:
        def log(self, *args):
            pass
    logger = Logger()

dataset = Dataset(cfg)
question_rnn = QuestionRNN(dataset, cfg)
if cfg.pre_trained_lstm:
    question_rnn.load('./language_model/pre-trained.pth')

env = gym.make(cfg.env_name)
env.seed(1)
np.random.seed(1)
torch.manual_seed(1)
random.seed(1)

env = OracleWrapper(env, syntax_error_reward=cfg.syntax_error_reward,
                    undefined_error_reward=cfg.undefined_error_reward,
                    ans_random=cfg.ans_random)
state_dim = env.observation_space['image'].shape
action_dim = env.action_space.n

# Store data for each run
runs_reward = []

# TODO - incorporate many runs for average results

print(f"========================== TRAINING - RUN {1 + 1:.0f}/{cfg.runs:.0f} ==========================")
# Agent
model = BrainNet(question_rnn)
agent = Agent(model, cfg.lr, cfg.lmbda, cfg.gamma, cfg.clip,
              cfg.value_param, cfg.entropy_act_param,
              cfg.policy_qa_param, cfg.entropy_qa_param)

# Agent defaults
# (self, model, learning_rate=0.001, lmbda=0.95, gamma=0.99, clip_param=0.2,
# value_param=1, entropy_act_param=0.01, policy_qa_param=1, entropy_qa_param=0.05):

print(agent.model)

_, train_reward = train(env, agent, logger, exploration=True, n_episodes=cfg.N_eps,
                        log_interval=cfg.train_log_interval, verbose=True)

# Store result for every run
runs_reward.append(train_reward)

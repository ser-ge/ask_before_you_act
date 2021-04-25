# Environment
import gym
import gym_minigrid

from agents.Agent import Agent
from models.brain_net import BrainNet
from oracle.oracle import OracleWrapper
from utils.Trainer import train
from utils.language import vocab
from language_model import Dataset, Model as QuestionRNN
from oracle.generator import gen_phrases
from typing import Callable
from dataclasses import dataclass, asdict

import wandb

@dataclass
class Config:
    epochs: int = 30
    batch_size : int = 256
    sequence_len: int = 10
    lstm_size : int = 128
    word_embed_dims : int = 128
    drop_out_prob : float = 0
    gen_phrases : Callable = gen_phrases
    hidden_dim : float = 32
    lr : float = 0.001
    gamma : float = 0.99
    lmbda : float = 0.95
    clip : float = 0.1
    entropy_param : float = 0.1
    value_param : float = 1
    N_eps : float = 2000
    train_log_interval : float = 25
    runs : float = 1
    env_name : str = "MiniGrid-Empty-5x5-v0"
    ans_random : bool = False


cfg = Config()

# %%

wandb.init(project='ask_before_you_act', config=asdict(cfg))


dataset = Dataset(cfg)
question_rnn = QuestionRNN(dataset, cfg)
question_rnn.load('./language_model/pre-trained.pth')

env = gym.make(cfg.env_name)
env = OracleWrapper(env, syntax_error_reward=-0.001, undefined_error_reward=-0.001, ans_random=cfg.ans_random)
env.seed(0)
state_dim = env.observation_space['image'].shape
action_dim = env.action_space.n


# Store data for each run
runs_reward = []

#TODO incoperate many runs for average results

print(f"========================== TRAINING - RUN {1 + 1:.0f}/{cfg.runs:.0f} ==========================")
# Agent
model = BrainNet(question_rnn)
agent = Agent(model, cfg.lr, cfg.gamma, cfg.clip, cfg.value_param, cfg.entropy_param, lmbda=cfg.lmbda)

print(agent.model)
_, train_reward = train(env, agent, wandb, exploration=True,
                           n_episodes=cfg.N_eps, log_interval=cfg.train_log_interval,
                           verbose=True)

# store result for every run
runs_reward.append(train_reward)









# Environment
import random

import gym
import gym_minigrid
import torch
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from agents.Agent import Agent
from agents.AgentMem import AgentMem
from models.brain_net import BrainNet
from models.brain_net_mem import BrainNetMem
from oracle.oracle import OracleWrapper
from utils.Trainer import train
from utils.TrainerMem import train as train_mem


from language_model import Dataset, Model as QuestionRNN
from oracle.generator import gen_phrases
from typing import Callable
from dataclasses import dataclass, asdict

import wandb

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
    N_eps: float = 250
    train_log_interval: float = 5
    env_name: str = "MiniGrid-Empty-5x5-v0"  # "MiniGrid-MultiRoom-N2-S4-v0" "MiniGrid-Empty-5x5-v0"
    ans_random: bool = False
    undefined_error_reward: float = -0.1
    syntax_error_reward: float = -0.2
    pre_trained_lstm: bool = True
    use_seed: bool = False
    seed: int = 1
    use_mem: bool = True


def run_experiment(USE_WANDB, **kwargs):
    cfg = Config(**kwargs)

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

    if cfg.use_seed:
        env.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        random.seed(cfg.seed)

    env = OracleWrapper(env, syntax_error_reward=cfg.syntax_error_reward,
                        undefined_error_reward=cfg.undefined_error_reward,
                        ans_random=cfg.ans_random)

    # Agent
    if cfg.use_mem:
        model = BrainNetMem(question_rnn)
        agent = AgentMem(model, cfg.lr, cfg.lmbda, cfg.gamma, cfg.clip,
                      cfg.value_param, cfg.entropy_act_param,
                      cfg.policy_qa_param, cfg.entropy_qa_param)

        _, train_reward = train_mem(env, agent, logger, exploration=True, n_episodes=cfg.N_eps,
                                log_interval=cfg.train_log_interval, verbose=True)

    else:
        model = BrainNet(question_rnn)
        agent = Agent(model, cfg.lr, cfg.lmbda, cfg.gamma, cfg.clip,
                      cfg.value_param, cfg.entropy_act_param,
                      cfg.policy_qa_param, cfg.entropy_qa_param)

        _, train_reward = train(env, agent, logger, exploration=True, n_episodes=cfg.N_eps,
                                log_interval=cfg.train_log_interval, verbose=True)

    return train_reward


def plot_experiment(runs_reward, total_runs):
    sns.set()

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111)

    window = 10
    avg = pd.DataFrame(np.array(runs_reward)).T.rolling(window).mean().T

    df_reward_v2 = pd.DataFrame(avg).melt()
    sns.lineplot(ax=ax, x='variable', y='value', data=df_reward_v2)
    ax.set_title(f"Reward training curve over {total_runs} runs")
    ax.set_ylabel(f"{window} episode moving average of mean agent\'s reward")
    ax.set_xlabel("Episodes")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Store data for each run
    runs_reward = []
    total_runs = 3
    for runs in range(total_runs):
        print(f"========================== TRAINING - RUN {1 + runs:.0f}/{total_runs:.0f} ==========================")
        train_reward = run_experiment(True)

        # Store result for every run
        runs_reward.append(train_reward)

    plot_experiment(runs_reward, total_runs)




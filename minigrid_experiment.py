# Environment
import random

import gym
import gym_minigrid
import torch
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from agents.BaselineAgent import PPOAgent
from agents.Agent import Agent, AgentMem

from models.BaselineModel import BaselineCNN
from models.brain_net import BrainNet, BrainNetMem

from oracle.oracle import OracleWrapper
from utils.Trainer import train
from utils.BaselineTrain import GAEtrain

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
    N_eps: float = 500
    train_log_interval: float = 25
    # env_name: str = "MiniGrid-Empty-8x8-v0"
    env_name: str = "MiniGrid-Empty-5x5-v0"
    ans_random: bool = False
    undefined_error_reward: float = -0.1
    syntax_error_reward: float = -0.2
    pre_trained_lstm: bool = True
    use_seed: bool = False
    seed: int = 1
    use_mem: bool = False
    baseline: bool = False


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = "cpu"


def run_experiment(USE_WANDB, **kwargs):
    cfg = Config(**kwargs)

    if USE_WANDB:
        run = wandb.init(project='ask_before_you_act', config=asdict(cfg))
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
    if cfg.baseline:
        model = BaselineCNN()
        agent = PPOAgent(model, cfg.lr, cfg.lmbda, cfg.gamma, cfg.clip,
                         cfg.value_param, cfg.entropy_act_param)

        _, train_reward = GAEtrain(env, agent, logger, n_episodes=cfg.N_eps,
                                   log_interval=cfg.train_log_interval, verbose=True)

    elif cfg.use_mem:
        model = BrainNetMem(question_rnn)
        agent = AgentMem(model, cfg.lr, cfg.lmbda, cfg.gamma, cfg.clip,
                      cfg.value_param, cfg.entropy_act_param,
                      cfg.policy_qa_param, cfg.entropy_qa_param)

        _, train_reward = train(env, agent, logger, memory=cfg.use_mem, n_episodes=cfg.N_eps,
                                log_interval=cfg.train_log_interval, verbose=True)


    else:
        model = BrainNet(question_rnn)
        agent = Agent(model, cfg.lr, cfg.lmbda, cfg.gamma, cfg.clip,
                      cfg.value_param, cfg.entropy_act_param,
                      cfg.policy_qa_param, cfg.entropy_qa_param)

        _, train_reward = train(env, agent, logger, memory=cfg.use_mem, n_episodes=cfg.N_eps,
                                log_interval=cfg.train_log_interval, verbose=True)

    if USE_WANDB:
        run.finish()
    return train_reward


def plot_experiment(averaged_data, total_runs, window=25):

    avg_rnd = averaged_data['Random Noise'].rolling(window).mean()
    avg_good = averaged_data['Actual Information'].rolling(window).mean()
    advantage = pd.Series(avg_rnd - avg_good)
    plt.style.use('dark_background')
    fig, axs = plt.subplots(2, 1, sharex='all')
    fig.tight_layout()

    axs[0].plot(avg_rnd,color='red')
    axs[0].plot(avg_good,color='green')
    axs[0].set_title(f"Reward training curves, smoothed over {total_runs} runs")
    axs[0].set_xlabel("Episodes")
    axs[0].set_ylabel(f"{window} ep moving avg of mean agent reward")

    axs[1].plot(advantage,color='blue')
    axs[1].set_title("Advantage of QA Agent")
    axs[1].set_xlabel("Episodes")

    plt.show()
    fig.savefig("./figures/figure_run" + str(total_runs) + signature + ".png")


if __name__ == "__main__":
    # Store data for each run
    cfg = Config()
    signature = str(random.randint(10000, 90000))
    runs_reward = []
    total_runs = 5
    data_to_be_averaged = np.zeros([cfg.N_eps,total_runs])
    averaged_data = pd.DataFrame(columns=['Random Noise','Actual Information'])

    column_number = 0


    for ans_random in (True, False):
        for runs in range(total_runs):
            print(f"================= RUN {1 + runs:.0f}/{total_runs:.0f} || RND. ANS - {ans_random} =================")
            train_reward = run_experiment(False, ans_random=ans_random)
            # you set a 'total_runs' parameter above
            # you then will take an average of the rewards achieved across these runs
            # i.e. you'll take the mean over the x axis of the rewards series..
            data_to_be_averaged[:,runs] = train_reward

        # then here you just fill in the data frame
        averaged_data.iloc[:,column_number] = data_to_be_averaged.mean(axis=1)
        # increment column number by 1 so that you then fill in the next column of the dataframe
        column_number += 1

    print('Runs Complete!')
    print(averaged_data)
    plot_experiment(averaged_data, total_runs)
    np.save("./data/runs_reward" + str(total_runs) + signature + ".npy", runs_reward)




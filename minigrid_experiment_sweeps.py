# Environment
import random
from pathlib import Path
import time
import gym
import gym_minigrid
import torch
import numpy as np


# %%
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# %%
from tqdm import tqdm

from agents.BaselineAgent import PPOAgentMem
from agents.Agent import Agent, AgentMem

from models.BaselineModel import BaselineMem
from models.brain_net import BrainNet, BrainNetMem

from oracle.oracle import OracleWrapper
from utils.Trainer import traintest
from utils.BaselineTrain import GAEtrain

from language_model import Dataset, Model as QuestionRNN

from oracle.generator import gen_phrases
from typing import Callable
from dataclasses import dataclass, asdict

import wandb

from itertools import product

device = "cpu"

RUNS_PATH = Path('./data')
USE_WANDB = False

@dataclass
class Config:
    "Default Arguments for experiments - Overide in sweep config"
    epochs: int = 30
    batch_size: int = 256
    sequence_len: int = 10
    lstm_size: int = 128
    word_embed_dims: int = 128
    drop_out_prob: float = 0
    gen_phrases: Callable = gen_phrases
    hidden_dim: float = 32
    lr: float = 0.0005
    gamma: float = 0.99
    lmbda: float = 0.95
    clip: float = 0.1
    entropy_act_param: float = 0.1
    value_param: float = 1
    policy_qa_param: float = 1
    entropy_qa_param: float = 0.05
    N_eps: float = 5
    train_log_interval: float = 50
    # env_name: str = "MiniGrid-Empty-8x8-v0"
    env_name: str = "MiniGrid-Empty-8x8-v0"
    ans_random: bool = False
    undefined_error_reward: float = -0.1
    syntax_error_reward: float = -0.2
    pre_trained_lstm: bool = True
    use_seed: bool = False
    seed: int = 1
    use_mem: bool = False
    baseline: bool = False



sweep_config = {
    "name" : "8 by 8 sweeep true false",
    "method": "bayes",
    "metric": {"name": "avg_reward_episodes", "goal": "maximize"},
    'parameters':
    {
    "lr": {
        "value": 0.001
    },
    "clip": {
        "value": 0.11382609211422028
    },
    "lmbda": {
        "value": 0.95
    },
    "env_name": {
        "value": "MiniGrid-Empty-8x8-v0"
    },
    "ans_random": {
        "values": [True, False]
    },
    "value_param": {
        "value": 0.8210391931653159
    },
    "policy_qa_param": {
        "value": 0.507744927219129
    },
    "entropy_qa_param": {
        "value": 0.28267454781905166
    },
    "entropy_act_param": {
        "value": 0.08081028521575984
    },
    "syntax_error_reward": {
        "value": -0.2
    },
    "undefined_error_reward": {
        "value": -0.1
    }
} }






class Logger:
    def log(self, *args):
        pass

def gen_configs(sweep_config):

    params = sweep_config['parameters']

    configs = []
    sweep_params = {}
    fixed_params = []


    for param in params:
        if 'values' in params[param].keys():
            sweep_params[param]  = params[param]['values']
        else:
            fixed_params.append(param)

    base_config = {param : params[param]['value'] for param in fixed_params}

    for param_values in product(*list(sweep_params.values())):
        cfg = base_config.copy()
        for i, param in enumerate(sweep_params.keys()):
            cfg[param] = param_values[i]

        configs.append(cfg)

    return configs




def run_experiments(sweep_config, num_runs=2, runs_path=RUNS_PATH):

    save_path = runs_path / Path('run_results_' + str(time.asctime()) + '.p')

    total_runs = 10

    configs = gen_configs(sweep_config)

    if len(configs) > 1 and USE_WANDB:
        sweep_id = wandb.sweep(sweep_config, project="ask_before_you_act")
        wandb.agent(sweep_id, function=run_experiment)
        return

    else:

        experiments = [{'config' : cfg, 'train_rewards':[]} for cfg in configs]

        print(f"Total of {len(experiments)} experiments collected for {num_runs} runs each")

        for i, exp in enumerate(experiments):
            config = Config(**exp['config'])
            print(config)
            print(f"Running Experiment {i}")
            for run in tqdm(range(num_runs)):
                train_reward = run_experiment(config)
                exp['train_rewards'].append(train_reward)

        pickle.dump(experiments, open(save_path, 'wb'))
        print(f"Run results saved to {save_path}")
        return experiments


def load_exp_results(file_name):
    return pickle.load(open(file_name, 'rb'))



def run_experiment(cfg=None):

    if cfg is None:
        cfg = Config()

    dataset = Dataset(cfg)
    question_rnn = QuestionRNN(dataset, cfg)


    if cfg.pre_trained_lstm:
        question_rnn.load('./language_model/pre-trained.pth')

    if USE_WANDB:
        run = wandb.init(project='ask_before_you_act', config=asdict(cfg))
        logger = wandb
    else:
        logger = Logger()


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
        model = BaselineMem()
        agent = PPOAgentMem(model, cfg.lr, cfg.lmbda, cfg.gamma, cfg.clip,
                         cfg.value_param, cfg.entropy_act_param)

        _, train_reward = GAEtrain(env, agent, logger, n_episodes=cfg.N_eps,
                                   log_interval=cfg.train_log_interval, verbose=True)
        #TODO fix GAE Train

    elif cfg.use_mem:
        model = BrainNetMem(question_rnn)
        agent = AgentMem(model, cfg.lr, cfg.lmbda, cfg.gamma, cfg.clip,
                      cfg.value_param, cfg.entropy_act_param,
                      cfg.policy_qa_param, cfg.entropy_qa_param)

        _, train_reward = traintest(env, agent, logger, memory=cfg.use_mem, n_episodes=cfg.N_eps,
                                log_interval=cfg.train_log_interval, verbose=True)


    else:
        model = BrainNet(question_rnn)
        agent = Agent(model, cfg.lr, cfg.lmbda, cfg.gamma, cfg.clip,
                      cfg.value_param, cfg.entropy_act_param,
                      cfg.policy_qa_param, cfg.entropy_qa_param)

        _, train_reward = traintest(env, agent, logger, memory=cfg.use_mem, n_episodes=cfg.N_eps,
                                log_interval=cfg.train_log_interval, verbose=True)

    if USE_WANDB: run.finish()

    return train_reward

# %%

def plot_experiment(runs_reward, total_runs, window=25):
    sns.set()

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111)

    avg_rnd = pd.DataFrame(np.array(runs_reward[:total_runs])).T.rolling(window).mean().T
    avg_good = pd.DataFrame(np.array(runs_reward[total_runs:-total_runs])).T.rolling(window).mean().T
    avg_base = pd.DataFrame(np.array(runs_reward[-total_runs:])).T.rolling(window).mean().T

    reward_rnd = pd.DataFrame(avg_rnd).melt()
    reward_good = pd.DataFrame(avg_good).melt()
    reward_base = pd.DataFrame(avg_base).melt()

    sns.lineplot(ax=ax, x='variable', y='value', data=reward_rnd, legend='brief', label="Random")
    sns.lineplot(ax=ax, x='variable', y='value', data=reward_good, legend='brief', label="Good")
    sns.lineplot(ax=ax, x='variable', y='value', data=reward_base, legend='brief', label="Baseline")

    ax.set_title(f"Reward training curve over {total_runs} runs")
    ax.set_ylabel(f"{window} episode moving average of mean agent\'s reward")
    ax.set_xlabel("Episodes")
    plt.tight_layout()
    plt.show()
    fig.savefig("./figures/figure_run" + str(total_runs) + signature + ".png")



if __name__ == '__main__':
    run_experiments(sweep_config)

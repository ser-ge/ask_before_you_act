# Environment
import random

import gym
import gym_minigrid
import torch
import numpy as np

import time
from itertools import product
from tqdm import tqdm

import math
import pprint
import pickle

from pathlib import Path
from agents.BaselineAgent import BaselineAgent, BaselineAgentExpMem
from agents.BrainAgent import Agent, AgentMem, AgentExpMem

from models.BaselineModel import BaselineModel, BaselineModelExpMem
from models.BrainModel import BrainNet, BrainNetMem, BrainNetExpMem

from oracle.oracle import OracleWrapper
from utils.Trainer import train_test

from language_model import Dataset, Model as QuestionRNN
from oracle.generator import gen_phrases
from typing import Callable
from dataclasses import dataclass, asdict

import wandb

@dataclass
class Config:
    train: bool = True
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
    train_episodes: float = 3000
    test_episodes: float = 10
    train_log_interval: float = 50
    test_log_interval: float = 1
    # env_name: str = "MiniGrid-Empty-8x8-v0"
    env_name: str = "MiniGrid-Empty-5x5-v0"
    ans_random: float = 0
    undefined_error_reward: float = 0
    syntax_error_reward: float = -0.2
    defined_q_reward : float = 0.2
    pre_trained_lstm: bool = True
    use_seed: bool = False
    seed: int = 1
    use_mem: bool = True
    exp_mem: bool = True
    baseline: bool = False

default_config = Config()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = "cpu"

USE_WANDB = True
NUM_RUNS = 2
RUNS_PATH = Path('./data')


sweep_config = {
    "name" : "MiniGrid-KeyCorridorS6R3-v0",
    "method": "bayes",
    "metric": {"name": "eps_reward", "goal": "maximize"},

    "parameters": {
        # "env_name" : {
        #     'value' : 'MiniGrid-KeyCorridorS3R1-v0'
        #     },

        "env_name" : {
            'value' : 'MiniGrid-MultiRoom-N2-S4-v0'
            },

        "entropy_qa_param": {
            "distribution": "uniform",
            "min": 0,
            "max": 0.2,
        },
        "entropy_act_param": {
            "distribution": "uniform",
            "min": 0,
            "max": 0.2,
        },
        "lr": {
            "distribution": "log_uniform",
            "min": math.log(1e-7),
            "max": math.log(0.1),
        },
        "value_param": {
            "distribution": "uniform",
            "min": 0.5,
            "max": 1,
        },
        "policy_qa_param": {
            "distribution": "uniform",
            "min": 0.5,
            "max": 1,
        },
        # "undefined_error_reward": {
        #     "distribution": "uniform",
        #     "min": -1,
        #     "max": 0,
        # },
        # "syntax_error_reward": {
        #     "distribution": "uniform",
        #     "min": -1,
        #     "max": 0,
        # },
        "clip": {
            "distribution": "uniform",
            "min": 0.05,
            "max": 0.3,
        },
        "ans_random" : {
            'value' :  0
            },
    }
}


sweep_config_8_8 = {
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
        "values": [0, 0.5, 1]
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



def run_experiments(configs=sweep_config, num_runs=NUM_RUNS, runs_path=RUNS_PATH):

    """
    pickle.load(open('data/run_results_Thu Apr 29 13:14:39 2021.p', 'rb'))
    """

    save_path = runs_path / Path('run_results_' + str(time.asctime()) + '.p')

    if USE_WANDB:
        sweep_id = wandb.sweep(configs, project="ask_before_you_act")
        wandb.agent(sweep_id, function=run_experiment)
        return

    else:
        configs = gen_configs(configs)
        experiments = [{'config' : cfg, 'train_rewards':[], 'test_rewards' : []} for cfg in configs]

        print(f"Total of {len(experiments)} experiments collected for {num_runs} runs each")

        for i, exp in enumerate(experiments):
            config = Config(**exp['config'])
            pprint.pprint(asdict(config))
            print(f"Running Experiment {i}")
            for run in tqdm(range(num_runs)):
                train_reward, test_reward = run_experiment(config)
                exp['train_rewards'].append(train_reward)
                exp['test_rewards'].append(test_reward)

        pickle.dump(experiments, open(save_path, 'wb'))
        print(f"Run results saved to {save_path}")
        return experiments




def run_experiment(cfg=default_config):


    dataset = Dataset(cfg)
    question_rnn = QuestionRNN(dataset, cfg)

    if USE_WANDB:
        run = wandb.init(project='ask_before_you_act', config=asdict(cfg))
        logger = wandb
        cfg = wandb.config
    else:

        logger = Logger()

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
                        defined_q_reward = cfg.defined_q_reward,
                        ans_random=cfg.ans_random)

    # Agent
    agent = set_up_agent(cfg, question_rnn)

    # Train
    train_reward = train_test(env, agent, cfg, logger, n_episodes=cfg.train_episodes,
                              log_interval=cfg.train_log_interval, train=True, verbose=True)

    test_reward = train_test(env, agent, cfg, logger, n_episodes=cfg.test_episodes,
                              log_interval=cfg.test_log_interval, train=False, verbose=True)

    if USE_WANDB:run.finish()
    return train_reward, test_reward


def plot_experiment(averaged_data, total_runs, window=25):

    # avg_rnd = averaged_data['Random Noise'].rolling(window).mean()
    # avg_good = averaged_data['Actual Information'].rolling(window).mean()
    advantage = pd.Series(averaged_data.iloc[:,0] - averaged_data.iloc[:,-1])
    plt.style.use('default')
    fig, axs = plt.subplots(2, 1, sharex='all')
    fig.tight_layout()

    axs[0].plot(averaged_data.rolling(window).mean())
    # axs[0].plot(avg_good,color='green')
    axs[0].set_title(f"Reward training curves, smoothed over {total_runs} runs")
    axs[0].set_xlabel("Episodes")
    axs[0].set_ylabel(f"{window} ep moving avg of mean agent reward")
    axs[0].legend(averaged_data.columns)

    axs[1].plot(advantage,color='blue')
    axs[1].set_title("Advantage of no random over random Agent")
    axs[1].set_xlabel("Episodes")
    plt.show()
    # fig.savefig("./figures/figure_run" + str(total_runs) + signature + ".png")


def set_up_agent(cfg, question_rnn):
    if cfg.baseline:
        if cfg.use_mem:
            model = BaselineModelExpMem()
            agent = BaselineAgentExpMem(model, cfg.lr, cfg.lmbda, cfg.gamma, cfg.clip,
                             cfg.value_param, cfg.entropy_act_param)

        else:
            model = BaselineModel()
            agent = BaselineAgent(model, cfg.lr, cfg.lmbda, cfg.gamma, cfg.clip,
                                        cfg.value_param, cfg.entropy_act_param)

    else:
        if cfg.use_mem and cfg.exp_mem:
            model = BrainNetExpMem(question_rnn)
            agent = AgentExpMem(model, cfg.lr, cfg.lmbda, cfg.gamma, cfg.clip,
                          cfg.value_param, cfg.entropy_act_param,
                          cfg.policy_qa_param, cfg.entropy_qa_param)

        elif cfg.use_mem and not cfg.exp_mem:
            model = BrainNetMem(question_rnn)
            agent = AgentMem(model, cfg.lr, cfg.lmbda, cfg.gamma, cfg.clip,
                                cfg.value_param, cfg.entropy_act_param,
                                cfg.policy_qa_param, cfg.entropy_qa_param)

        else:
            model = BrainNet(question_rnn)
            agent = Agent(model, cfg.lr, cfg.lmbda, cfg.gamma, cfg.clip,
                          cfg.value_param, cfg.entropy_act_param,
                          cfg.policy_qa_param, cfg.entropy_qa_param)
    return agent


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


if __name__ == "__main__":
    run_experiments()

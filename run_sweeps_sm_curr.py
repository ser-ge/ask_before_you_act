# Environment
import random

import gym
import gym_minigrid
import torch
import numpy as np
import time
from itertools import product
from tqdm import tqdm

import yaml
import math
import pprint
import pickle

from pathlib import Path
from agents.BaselineAgent import BaselineAgent, BaselineAgentExpMem
from agents.BrainAgent import Agent, AgentMem, AgentExpMem

from models.BaselineModel import BaselineModel, BaselineModelExpMem
from models.BrainModel import BrainNet, BrainNetMem, BrainNetExpMem
from models.FilmModel import FilmNet

from oracle.oracle import OracleWrapper
from utils.Trainer import train_test

from language_model import Dataset, Model as QuestionRNN
from oracle.generator import gen_phrases
from typing import Callable
from dataclasses import dataclass, asdict

import wandb
# %%

@dataclass
class Config:
    train: bool = True
    epochs: int = 30
    batch_size: int = 256
    sequence_len: int = 10
    lstm_size: int = 128
    word_embed_dims: int = 128
    drop_out_prob: float = 0
    hidden_dim: float = 32

    lr: float = 0.0005
    gamma: float = 0.99
    lmbda: float = 0.95
    clip: float = 0.2
    entropy_act_param: float = 0.1
    value_param: float = 1

    policy_qa_param: float = 0.25
    advantage_qa_param: float = 0.25
    entropy_qa_param: float = 0.05

    train_episodes: float = 3000
    test_episodes: float = 0

    train_log_interval: float = 3
    test_log_interval: float = 1
    log_questions: bool = False

    train_env_name: str =  "MiniGrid-FourRooms-v0"
    test_env_name: str = "MiniGrid-FourRooms-v0"

    ans_random: float = 0

    undefined_error_reward: float = 0
    syntax_error_reward: float = -0.2
    defined_q_reward: float = 0.2
    defined_q_reward_test : float = 0

    pre_trained_lstm: bool = True

    use_mem: bool = True
    use_seed: bool = False
    exp_mem: bool = True

    film: bool = True
    baseline: bool = True
    wandb: bool = True
    notes: str = ""


# %%
def load_yaml_config(path_to_yaml):
    try:
        with open (path_to_yaml, 'r') as file:
            config = yaml.safe_load(file)
            return config
    except Exception as e:
        print('Error reading the config file')


yaml_config = load_yaml_config("./config.yaml")
yaml_config = Config(**yaml_config)

pprint.pprint(yaml_config)

device = "cpu"



sweep_config = {
        "name" : f"Sweep: baseline: {yaml_config.baseline}, film : {yaml_config.film} env: {yaml_config.train_env_name}",
    "method": "random",
    "metric": {"name": "train/avg_reward_episodes", "goal": "maximize"},

    "parameters" :
        dict() }



def run_sweep(configs=sweep_config):
    """
    pickle.load(open('data/run_results_Thu Apr 29 13:14:39 2021.p', 'rb'))
    """
    sweep_id = wandb.sweep(configs, project="ask_before_you_act")
    wandb.agent(sweep_id, function=run_experiment)

class Logger:
    def log(self, *args):
        pass

def run_experiment(cfg=yaml_config):
    dataset = Dataset(cfg)
    question_rnn = QuestionRNN(dataset, cfg)

    if cfg.wandb:
        run = wandb.init(project='ask_before_you_act', config=asdict(cfg))
        logger = wandb
        cfg = wandb.config
    else:
        logger = Logger()

    if cfg.pre_trained_lstm:
        question_rnn.load('./language_model/pre-trained.pth')

    env_train = gym.make(cfg.train_env_name)
    env_test = gym.make(cfg.test_env_name)

    if cfg.use_seed:
        env_test.seed(cfg.seed)
        env_train.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        random.seed(cfg.seed)

    env_train = OracleWrapper(env_train, syntax_error_reward=cfg.syntax_error_reward,
                              undefined_error_reward=cfg.undefined_error_reward,
                              defined_q_reward=cfg.defined_q_reward,
                              ans_random=cfg.ans_random)

    env_test = OracleWrapper(env_test, syntax_error_reward=cfg.syntax_error_reward,
                              undefined_error_reward=cfg.undefined_error_reward,
                              defined_q_reward=cfg.defined_q_reward_test,
                              ans_random=cfg.ans_random)

    # Agent
    agent = set_up_agent(cfg, question_rnn)

    # Train
    train_reward = train_test(env_train, agent, cfg, logger, n_episodes=cfg.train_episodes,
                              log_interval=cfg.train_log_interval, train=True, verbose=True, test_env=False)

    # Test

    if cfg.test_episodes:
        test_reward = train_test(env_test, agent, cfg, logger, n_episodes=cfg.test_episodes,
                                  log_interval=cfg.train_log_interval, train=True, verbose=True, test_env=True)


    if cfg.wandb:
        run.finish()



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
        if cfg.use_mem and cfg.exp_mem and cfg.film:
            model = FilmNet(question_rnn)
            agent = AgentExpMem(model, cfg.lr, cfg.lmbda, cfg.gamma, cfg.clip,
                                cfg.value_param, cfg.entropy_act_param,
                                cfg.policy_qa_param, cfg.advantage_qa_param,
                                cfg.entropy_qa_param)

        elif cfg.use_mem and cfg.exp_mem:
            model = BrainNetExpMem(question_rnn)
            agent = AgentExpMem(model, cfg.lr, cfg.lmbda, cfg.gamma, cfg.clip,
                                cfg.value_param, cfg.entropy_act_param,
                                cfg.policy_qa_param, cfg.advantage_qa_param,
                                cfg.entropy_qa_param)

        elif cfg.use_mem and not cfg.exp_mem:
            model = BrainNetMem(question_rnn)
            agent = AgentMem(model, cfg.lr, cfg.lmbda, cfg.gamma, cfg.clip,
                             cfg.value_param, cfg.entropy_act_param,
                             cfg.policy_qa_param, cfg.advantage_qa_param,
                             cfg.entropy_qa_param)

        else:
            model = BrainNet(question_rnn)
            agent = Agent(model, cfg.lr, cfg.lmbda, cfg.gamma, cfg.clip,
                          cfg.value_param, cfg.entropy_act_param,
                          cfg.policy_qa_param, cfg.advantage_qa_param,
                          cfg.entropy_qa_param)
    return agent




if __name__ == "__main__":
    if yaml_config.wandb:
        run_sweep()
    else:
        run_experiment()

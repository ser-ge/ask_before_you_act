import torch
import numpy as np
from language_model import Dataset, Model as QuestionRNN
from oracle.oracle import OracleWrapper
from utils.Trainer import train_test
from utils import default_config, load_agent, save_agent, set_up_agent
import wandb
import random
import gym
from dataclasses import asdict
import gym_minigrid

class Logger:
    def log(self, *args):
        pass

def train(cfg):
    if cfg.wandb:
        run = wandb.init(project='ask_before_you_act', config=asdict(cfg))
        logger = wandb
        cfg = wandb.config
    else:

        logger = Logger()

    env = gym.make(cfg.train_env_name)

    if cfg.use_seed:
        env.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        random.seed(cfg.seed)

    env = OracleWrapper(env, syntax_error_reward=cfg.syntax_error_reward,
                        undefined_error_reward=cfg.undefined_error_reward,
                        defined_q_reward=cfg.defined_q_reward,
                        ans_random=cfg.ans_random)

    phrases = Dataset(cfg)
    question_rnn = QuestionRNN(phrases, cfg)

    if cfg.pre_trained_lstm:
        question_rnn.load('./language_model/pre-trained.pth')

    # Agent
    agent = set_up_agent(cfg, question_rnn)

    # Train
    reward = train_test(env, agent, cfg, logger, n_episodes=cfg.train_episodes,
                              log_interval=cfg.train_log_interval, train=True, verbose=True)

    save_agent(agent, cfg, cfg.name)

    if cfg.wandb: run.finish()

    return agent


if __name__ == "__main__":
    cfg= default_config
    cfg.name = 'agent-1'
    cfg.train_env_name='MiniGrid-Empty-8x8-v0'
    cfg.train_episodes=100

    train(default_config)
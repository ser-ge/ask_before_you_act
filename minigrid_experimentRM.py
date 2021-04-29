# Environment
import random

import gym
import gym_minigrid
import torch
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from agents.BaselineAgentRM import BaselineAgent, BaselineAgentExpMem
from agents.AgentRM import Agent, AgentMem, AgentExpMem

from models.BaselineModelRM import BaselineModel, BaselineModelExpMem
from models.brain_netRM import BrainNet, BrainNetMem, BrainNetExpMem

from oracle.oracle import OracleWrapper
from utils.TrainerRM import train_test

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
    train_episodes: float = 500
    test_episodes: float = 10
    train_log_interval: float = 50
    test_log_interval: float = 1
    # env_name: str = "MiniGrid-Empty-8x8-v0"
    env_name: str = "MiniGrid-Empty-5x5-v0"
    ans_random: float = 0
    undefined_error_reward: float = -0.1
    syntax_error_reward: float = -0.2
    pre_trained_lstm: bool = True
    use_seed: bool = False
    seed: int = 1
    use_mem: bool = False
    exp_mem: bool = False
    baseline: bool = True


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = "cpu"

default_config = Config()
USE_WANDB = False


class Logger:
    def log(self, *args):
        pass

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




def run_experiments(sweep_config, num_runs=2, runs_path=RUNS_PATH):

    save_path = runs_path / Path('run_results_' + str(time.asctime()) + '.p')


    if USE_WANDB:
        sweep_id = wandb.sweep(sweep_config, project="ask_before_you_act")
        wandb.agent(sweep_id, function=run_experiment)
        return

    else:
        configs = gen_configs(sweep_config)
        experiments = [{'config' : cfg, 'train_rewards':[], 'test_rewards' : []} for cfg in configs]

        print(f"Total of {len(experiments)} experiments collected for {num_runs} runs each")

        for i, exp in enumerate(experiments):
            config = Config(**exp['config'])
            print(config)
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
                        ans_random=cfg.ans_random)

    # Agent
    agent = set_up_agent(cfg, question_rnn)

    # Train
    train_reward = train_test(env, agent, cfg, logger, n_episodes=cfg.train_episodes,
                              log_interval=cfg.train_log_interval, train=True, verbose=True)

    test_reward = train_test(env, agent, cfg, logger, n_episodes=cfg.test_episodes,
                              log_interval=cfg.test_log_interval, train=True, verbose=True)


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
    # Store data for each run
    cfg = Config()
    signature = str(random.randint(10000, 90000))
    # runs_reward = []
    total_runs = 5
    data_to_be_averaged = np.zeros([cfg.train_episodes, total_runs])
    epsilon_range = np.linspace(0, 1, 5)
    averaged_data = pd.DataFrame(columns=[i for i in epsilon_range])
    column_number = 0


    for epsilon in epsilon_range:
        for runs in range(total_runs):
            print(f"================= RUN {1 + runs:.0f}/{total_runs:.0f} || RND. RandAnsEps- {epsilon} =================")
            train_reward = run_experiment(False, ans_random=epsilon, train=cfg.train)
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
    # np.save("./data/runs_reward" + str(total_runs) + signature + ".npy", averaged_data)

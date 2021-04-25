# Environment
import gym
import gym_minigrid

from agents.Agent import Agent
from ask_before_you_act.models.Policy import BrainNet
from oracle.oracle import OracleWrapper
from utils.Trainer import train
from utils.language import vocab
from language_model import Dataset, Model as QuestionRNN
from oracle.generator import gen_phrases
from typing import Callable
from dataclasses import dataclass

@dataclass
class Config:
    epochs: int = 30
    batch_size : int = 256
    sequence_len: int = 10
    lstm_size : int = 128
    word_embed_dims : int = 128
    drop_out_prob : float = 0
    gen_phrases : Callable = gen_phrases

cfg = Config()
dataset = Dataset(cfg)
question_rnn = QuestionRNN(dataset, cfg)
question_rnn.load('./language_model/pre-trained.pth')

# env = gym.make("MiniGrid-MultiRoom-N6-v0")
env = gym.make("MiniGrid-Empty-5x5-v0")
# MiniGrid-MultiRoom-N2-S4-v0, MiniGrid-Empty-5x5-v0
env = OracleWrapper(env, syntax_error_reward=-0.001, undefined_error_reward=-0.001)
env.seed(0)
state_dim = env.observation_space['image'].shape
action_dim = env.action_space.n

# Agent Params
hidden_dim = 32
# lr = 0.0001
lr = 0.001
gamma = 0.99
lmbda = 0.95
# clip = 0.2
clip = 0.1
# entropy_param = 0.01
entropy_param = 0.1
value_param = 1

N_eps = 2000
train_log_interval = 25

runs = 1

# Store data for each run
runs_reward = []

for i in range(runs):
    print(f"========================== TRAINING - RUN {i + 1:.0f}/{runs:.0f} ==========================")
    # Agent
    model = BrainNet(question_rnn)
    agent = Agent(model, lr, gamma, clip, value_param, entropy_param)

    print(agent.model)
    _, train_reward = train(env, agent, exploration=True,
                               n_episodes=N_eps, log_interval=train_log_interval,
                               verbose=True)

    # store result for every run
    runs_reward.append(train_reward)
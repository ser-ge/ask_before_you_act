from gym_minigrid.minigrid import Grid, Goal
from gym_minigrid import envs
import re
import gym
import gym_minigrid
import random
import torch
import numpy as np
from oracle.oracle import OracleWrapper

def make_env(env_name):
    empty_room_match = re.match(r"MiniGrid-Empty-Random-([0-9]+)x[0-9]+", env_name)
    if empty_room_match:
        env = EmptyRandomEnv(int(empty_room_match.group(1)))
    else:
        env = gym.make(env_name)

    return env


def make_oracle_envs(cfg):
    env_train = make_env(cfg.train_env_name)
    env_test = make_env(cfg.test_env_name)

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

    return env_train, env_test

class EmptyRandomEnv(envs.EmptyEnv):
    def __init__(self, size=20):
        super().__init__(size=size, agent_start_pos=None)

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal randomly
        self.place_obj(Goal())

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"
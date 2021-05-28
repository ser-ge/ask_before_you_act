import pprint
from utils import load_yaml_config
from utils.Trainer import train_test
from utils.agent import set_up_agent, load_agent, save_agent
from utils.env import make_oracle_envs
from dataclasses import asdict
import wandb
import argparse

parser = argparse.ArgumentParser(description='Run experiments')
parser.add_argument('-c', '--config',
                       metavar='config',
                       type=str,
                       default='./config.yaml',
                       help='the config path')

parser.add_argument('-n','--number-of-experiments',
                       metavar='number-of-experiments',
                       dest='number_of_experiments',
                       type=int,
                       default=10,
                       help='the number of experiments to run')

args = parser.parse_args()

def run_experiment(cfg):
    cfg.ans_random = False
    if cfg.wandb:
        run = wandb.init(project='ask_before_you_act', config=asdict(cfg), reinit=True)
        logger = wandb
    else:
        logger = None

    # Env
    env_train, env_test = make_oracle_envs(cfg)

    # Agent
    if cfg.load:
        agent = load_agent(cfg.name)
    else:
        agent = set_up_agent(cfg)

    if cfg.train_episodes:
        # Train
        train_reward = train_test(env_train, agent, cfg, logger, n_episodes=cfg.train_episodes,
                              log_interval=cfg.train_log_interval, train=True, verbose=True, test_env=False)
        save_agent(agent, cfg, cfg.name)

    # Test normal
    if cfg.test_episodes:
        test_reward = train_test(env_test, agent, cfg, logger, n_episodes=cfg.test_episodes,
                                  log_interval=cfg.train_log_interval, train=True, verbose=True, test_env=True)
        save_agent(agent, cfg, cfg.name + '-test')

    if cfg.wandb: run.finish()

    return train_reward, test_reward

def random_experiment(cfg):
    cfg.ans_random = True

    if cfg.wandb:
        run = wandb.init(project='ask_before_you_act', config=asdict(cfg), reinit=True)
        logger = wandb

    agent = load_agent(cfg.name)
    _, env_test = make_oracle_envs(cfg)

    # Test normal
    if cfg.test_episodes:
        test_reward = train_test(env_test, agent, cfg, logger, n_episodes=cfg.test_episodes,
                                 log_interval=cfg.train_log_interval, train=True, verbose=True, test_env=True)
        save_agent(agent, cfg, cfg.name + '-test_random')

    if cfg.wandb: run.finish()

if __name__ == "__main__":
    config_path = args.config
    cfg = load_yaml_config(config_path)
    print(f'Running {args.number_of_experiments} experiments')
    pprint.pprint(cfg)

    for i in range(args.number_of_experiments):
        run_experiment(cfg)
        random_experiment(cfg)
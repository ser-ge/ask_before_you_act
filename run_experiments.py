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
                       default='./main_config.yaml',
                       help='the config path')

parser.add_argument('-n','--number-of-experiments',
                       metavar='number-of-experiments',
                       dest='number_of_experiments',
                       type=int,
                       default=1,
                       help='the number of experiments to run')

parser.add_argument('-w','--wandb',
                        type=bool,
                        default=False,
                        dest='wandb',
                       help='log data to wandb, you must be logged into wandb')


parser.add_argument('-ansr','--ans_random',
                       metavar='number-of-experiments',
                       dest='number_of_experiments',
                       type=int,
                       default=0,
                       help='test on envrionmet with noisey oracle after training')


parser.add_argument('-v','--verbose',
                       metavar='number-of-experiments',
                       dest='verbose',
                       type=int,
                       default=100,
                       help='logging and printing interval')


parser.add_argument('--env_train',
                       dest='train_env_name',
                       type=str,
                       default="MiniGrid-MultiRoom-N2-S4-v0",
                       help='training eviroment, this runs first')


parser.add_argument('--env_test',
                       dest='test_env_name',
                       type=str,
                       default= "MiniGrid-MultiRoom-N4-S5-v0",
                       help='testing enviroment, for generalisation testing')

parser.add_argument('--episodes',
                       dest='epsisodes',
                       type=int,
                       default= 7500,
                       help='number of episodes for train and test env runs')

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
    else:
        logger = None

    agent = load_agent(cfg.name)
    _, env_test = make_oracle_envs(cfg)

    # Test normal
    if cfg.test_episodes:
        test_reward = train_test(env_test, agent, cfg, logger, n_episodes=cfg.test_episodes,
                                 log_interval=cfg.train_log_interval, train=True, verbose=True, test_env=True)
        save_agent(agent, cfg, cfg.name + '-test_random')

    if cfg.wandb: run.finish()

if __name__ == "__main__":
    args = parser.parse_args()
    config_path = args.config
    cfg = load_yaml_config(config_path)
    cfg.wandb = args.wandb
    cfg.train_log_interval = args.verbose
    cfg.test_log_interval = args.verbose
    cfg.train_env_name = args.train_env_name
    cfg.test_env_name = args.test_env_name
    cfg.test_episodes = cfg.train_episodes = args.epsisodes

    print(f'Running {args.number_of_experiments} experiments')
    pprint.pprint(cfg)

    for i in range(args.number_of_experiments):
        run_experiment(cfg)
        if not cfg.baseline:
            random_experiment(cfg)

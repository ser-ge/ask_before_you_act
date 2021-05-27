from dataclasses import dataclass

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
    entropy_act_param: float = 0.05
    value_param: float = 1
    policy_qa_param: float = 0.25
    advantage_qa_param: float = 0.25
    entropy_qa_param: float = 0.05
    train_episodes: float = 10000
    test_episodes: float = 5000
    train_log_interval: float = 5
    test_log_interval: float = 5
    train_env_name: str = "MiniGrid-MultiRoom-N2-S4-v0"
    test_env_name: str = "MiniGrid-MultiRoom-N4-S5-v0"
    ans_random: float = 0
    undefined_error_reward: float = 0
    syntax_error_reward: float = -0.2
    defined_q_reward: float = 0.2
    pre_trained_lstm: bool = True
    use_seed: bool = False
    seed: int = 1
    use_mem: bool = True
    exp_mem: bool = True
    baseline: bool = False
    wandb: bool = False

default_config = Config()
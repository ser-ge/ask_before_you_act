import base64
import glob
import io
from IPython.display import HTML
from IPython import display
from gym.wrappers import Monitor
from dataclasses import dataclass

from agents.BaselineAgent import BaselineAgentExpMem, BaselineAgent
from agents.BrainAgent import AgentExpMem, AgentMem, Agent
from models.BaselineModel import BaselineModelExpMem, BaselineModel
from models.BrainModel import BrainNetExpMem, BrainNetMem, BrainNet

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
    train_log_interval: float = 250
    test_log_interval: float = 100
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

def show_video():
    mp4list = glob.glob('video/*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        display.display(HTML(data='''<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")

def wrap_env_video_monitor(env):
    env = Monitor(env, './video', force=True)
    return env

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

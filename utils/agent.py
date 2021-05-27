from agents.BaselineAgent import BaselineAgentExpMem, BaselineAgent
from agents.BrainAgent import AgentExpMem, AgentMem, Agent
from models.BaselineModel import BaselineModelExpMem, BaselineModel
from models.BrainModel import BrainNetExpMem, BrainNetMem, BrainNet
from language_model import Dataset, Model as QuestionRNN
import utils

def save_agent(agent, cfg, name):
    model_dir = utils.get_model_dir(name)
    status = {
        "model_state": agent.model.state_dict(),
        "config": cfg,
    }
    utils.save_status(status, model_dir)

def load_agent(name):
    model_dir = utils.get_model_dir(name)
    config = utils.get_config(model_dir)
    dataset = Dataset(config)
    question_rnn = QuestionRNN(dataset, config)
    agent = set_up_agent(config, question_rnn)
    agent.model.load_state_dict(utils.get_model_state(model_dir))
    return agent

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

import yaml
import pandas as pd

config_path = '../config.yaml'

def yaml_jank(path_to_yaml):
    with open (path_to_yaml, 'r') as file:
        return yaml.safe_load(file)

config = yaml_jank(config_path)

cols = {'batch_size', 'sequence_len', 'lstm_size',
'word_embed_dims', 'drop_out_prob', 'hidden_dim', 'lr', 'gamma',
       'lmbda', 'clip', 'entropy_act_param', 'value_param', 'policy_qa_param',
       'advantage_qa_param', 'entropy_qa_param', 'undefined_error_reward', 'syntax_error_reward', 'defined_q_reward',
       'defined_q_reward_test', 'pre_trained_lstm', 'use_mem'}

a = pd.json_normalize(config)
# Latex
a[cols].T.to_latex(index=True)






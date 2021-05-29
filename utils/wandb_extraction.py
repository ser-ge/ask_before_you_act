import wandb
import pandas as pd

table_name = "ask_before_you_act"

api = wandb.Api()

runs = api.runs("rossmurphy/ask_before_you_act")

cols = ["lr", "gamma", "lmbd", "clip", "entropy_act_param", "value_param", "policy_qa_param", "advantage_qa_param", "entropy_qa_param", "train_episodes", "test_episodes", "train_log_interval", "test_log_interval", "log_questions", "train_env_name", "test_env_name", "ans_random", "undefined_error_reward", "syntax_error_reward", "defined_q_reward", "defined_q_reward_test", "pre_trained_lstm", "use_mem", "use_seed", "exp_mem", "baselin", "film", "wandb", "notes", "load", "name"]

summary_list = []
config_list = []
name_list = []
for run in runs:
    # run.summary are the output key/values like accuracy.  We call ._json_dict to omit large files
    summary_list.append(run.summary._json_dict)
    # run.config is the input metrics.  We remove special values that start with _.
    config_list.append({k:v for k,v in run.config.items() if not k.startswith('_')})
    # run.name is the name of the run.
    name_list.append(run.name)

summary_df = pd.DataFrame.from_records(summary_list)
config_df = pd.DataFrame.from_records(config_list)
name_df = pd.DataFrame({'name': name_list})
all_df = pd.concat([name_df, config_df,summary_df], axis=1)
all_df.to_csv("project.csv")

all_df.to_pickle(table_name+".pkl")

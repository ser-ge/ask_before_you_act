import wandb
import pandas as pd

# %% [markdown]
"""
Download all runs from wandb and make results table
Extract epochs with max val acc from each run and model config
create full results df
save results df to latex table (in txt file)
save results to pickle
"""
# %%
table_name = "q2_results_table_rm_latest"

api = wandb.Api()

def to_latex(df, name):
    df.to_latex(name + ".txt", index=True, column_format='l|r|r|r|r|', label=name)

def cols_to_percent(df, cols, inplace=True):
    if inplace:
        for col in cols:
            df[col] = pd.Series(["{0:.2f}%".format(val * 100) for val in df[col]], index = df.index)
        return df


runs = api.runs("ser-ge/fmnist_conv_net")


cols = ['val_acc',  'val_loss', 'test_acc','test_loss','train_acc', 'train_loss',
        'train_loss_no_reg', 'epoch','train_acc_no_reg']

config_cols = [
        ]
results = []

for run in runs:
    hist = run.history()

    # skip runs with missing cols
    if not set(cols).issubset(hist.columns):
        continue

    hist = hist[cols]

    hist = hist[hist.val_loss == min(hist.val_loss)].iloc[0]

    hist['name'] = run.name


    config = {k:v for k,v in run.config.items() if not k.startswith('_')}
    config = pd.Series(config)

    result = hist.append(config)
    results.append(result)



results_df = pd.concat(results, 1).transpose()

results_df = results_df.sort_values("val_loss", ascending=True).reset_index(inplace=False)

results_df.to_pickle(table_name+".pkl")

# drop cols
# results_df = results_df.drop(["index"])

percent_cols = [col for col in results_df.columns if col.endswith("acc")]

results_df = cols_to_percent(results_df, percent_cols)

to_latex(results_df, table_name)

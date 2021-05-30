import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ross_train_ans_random_0 = pd.read_csv('~/Downloads/ross/random_train.csv')
ross_train_ans_random_1 = pd.read_csv('~/Downloads/ross/non_random_train.csv')
ross_test_ans_random_0 = pd.read_csv('~/Downloads/ross/random_test.csv')
ross_test_ans_random_1 = pd.read_csv('~/Downloads/ross/non_random_test.csv')

thymo_train_ans_random_0 = pd.read_csv('~/Downloads/thymo/random_train.csv')
thymo_train_ans_random_1 = pd.read_csv('~/Downloads/thymo/non_random_train.csv')
thymo_test_ans_random_0 = pd.read_csv('~/Downloads/thymo/random_test.csv')
thymo_test_ans_random_1 = pd.read_csv('~/Downloads/thymo/non_random_test.csv')

train_ans_random_0 = pd.concat([ross_train_ans_random_0,thymo_train_ans_random_0],axis=1)
train_ans_random_1 = pd.concat([ross_train_ans_random_1,thymo_train_ans_random_1],axis=1)
test_ans_random_0 = pd.concat([ross_test_ans_random_0,thymo_test_ans_random_0],axis=1)
test_ans_random_1 = pd.concat([ross_test_ans_random_1,thymo_test_ans_random_1],axis=1)

negate = ['MIN','MAX']

dfs = [train_ans_random_0,train_ans_random_1, \
       test_ans_random_0,test_ans_random_1]

def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]

for i in dfs:
    i.drop(columns='Step')
    for col in i.columns:
        name = namestr(i, globals())[0]
        i.rename(columns = {col:str(name) + '_' + str(col)},inplace=True)


big_df = pd.concat([train_ans_random_1,train_ans_random_1, \
                    test_ans_random_0,test_ans_random_1],axis=1)

big_df.columns

def scrap_columns(df,list_to_scrap):
    lst = df.columns.to_list()
    for scrap in list_to_scrap:
        lst = list(filter(lambda k: scrap not in k, lst))
    return df[lst]






column_list = []
for columns in big_df.columns:
    column_list.append(columns)
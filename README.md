# Ask Before You Act



This repository contains the code for training and evaluation of our final NLP
project: "Ask Before You Act".

![Model Overview](/figures/model_overview.png)



## Abstract

Solving temporally-extended tasks is a challenge for most reinforcement
learning (RL) algorithms. We hypothesise that the use of natural language
is a critical tool to understand the hierarchical and relational nature of such
real-world tasks. Thus we investigate the ability of an RL agent to learn to ask
natural language questions as a tool to navigate its world and achieve greater
generalisation performance in new environments.  We do this by endowing this
agent with the ability of asking “yes-no”questions to an all-knowing Oracle.
This allows the agent to obtain guidance about the task at hand while limiting
the access to new information.  To study the emergence of such natural language
questions in the context of temporally-extended tasks we train this agent on
the Mini-Grid environments. We then transfer the trained agent on a different,
harder environment.  We observe a significant increase in
generalisation performance compared to a baseline agent unable to ask
questions. Through grounding the emergent natural language questions, the
agent can reason about the dynamics of its environment to the point that it can
ask new, relevant questions when deployed to a novel environmen.


## Installation


The required Python packages can be installed by running:

```shell
pip3 install --user --upgrade -r requirements.txt
```


## Trainning and Evaluation


The default configurations for each of the models can be found in the following files:

```
baseline_config.yaml
film_config.yaml
main_config.yaml
```

To train the main agent on the MiniGrid-MultiRoom-N2-S4-v0  and test
generalisation performance  on MiniGrid-MultiRoom-N4-S5-v0 you can run the
following command:


```shell
python run_experiments.py --env_train MiniGrid-MultiRoom-N2-S4-v0 --env_test -MiniGrid-MultiRoom-N4-S5-v0 -episodes 7500 --verbose 20 -c main_config.yaml
```




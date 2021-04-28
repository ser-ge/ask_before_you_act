import numpy as np
from collections import namedtuple

import torch
import wandb

Transition = namedtuple(
    "Transition",
    [
        "state",
        "answer",
        "hidden_q",
        "action",
        "reward",
        "reward_qa",
        "next_state",
        "log_prob_act",
        "log_prob_qa",
        "entropy_act",
        "entropy_qa",
        "done",
        "hidden_hist_mem",
        "cell_hist_mem",
        "next_hidden_hist_mem",
        "next_cell_hist_mem",
    ],
)

class TrainerTester:

    def __init__(self, env, agent, logger, n_episodes=1000,
          log_interval=50, memory=False, verbose=False):

        self.env = env
        self.agent = agent
        self.logger = logger
        self.n_episodes = n_episodes
        self.log_interval = log_interval
        self.memory = memory
        self.verbose = verbose

    def train(self,train=True):

        episode = 0

        episode_reward = []
        episode_qa_reward = []
        loss_history = []
        reward_history = []

        state = self.env.reset()['image']  # Discard other info
        step = 0

        avg_syntax_r = 0

        qa_pairs = []

        # Initialize random memory
        hist_mem = self.agent.init_memory()

        while episode < self.n_episodes:
            # Ask before you act
            question, hidden_q, log_prob_qa, entropy_qa = self.agent.ask(state, hist_mem[0])

            # Answer
            answer, reward_qa = self.env.answer(question)
            qa_pairs.append([question, str(answer), reward_qa])  # Storing
            answer = answer.decode()  # For passing vector to agent
            avg_syntax_r += 1 / self.log_interval * (reward_qa - avg_syntax_r)

            # Act
            action, log_prob_act, entropy_act = self.agent.act(state, answer, hidden_q)

            # Remember
            if self.memory:
                next_hist_mem = self.agent.remember(state, action, answer, hidden_q, hist_mem)
            else:
                next_hist_mem = self.agent.init_memory()

            # Step
            next_state, reward, done, _ = self.env.step(action)
            next_state = next_state['image']  # Discard other info

            # Store
            t = Transition(state, answer, hidden_q, action, reward, reward_qa, next_state,
                           log_prob_act.item(), log_prob_qa, entropy_act.item(), entropy_qa, done,
                           hist_mem[0], hist_mem[1], next_hist_mem[0], next_hist_mem[1])
            self.agent.store(t)

            # Advance
            state = next_state
            hist_mem = next_hist_mem  # Random hist_mem if nto using memory
            step += 1

            # Logging
            episode_reward.append(reward)
            episode_qa_reward.append(reward_qa)

            if done:
                # Update
                episode_loss = self.agent.update(train=train)

                # Reset episode
                state = self.env.reset()['image']  # Discard other info
                hist_mem = self.agent.init_memory()  # Initialize memory
                step = 0

                loss_history.append(episode_loss)
                reward_history.append(sum(episode_reward))

                self.logger.log(
                    {
                        "questions": wandb.Table(data=qa_pairs, columns=["Question", "Answer", "Reward"]),
                        "eps_reward": sum(episode_reward),
                        "avg_reward_qa": sum(episode_qa_reward) / len(episode_qa_reward),
                        "loss": episode_loss
                    }
                )

                episode_reward = []
                episode_qa_reward = []
                qa_pairs = []

                episode += 1

                if episode % self.log_interval == 0:
                    if self.verbose:
                        avg_R = np.sum(reward_history[-self.log_interval:]) / self.log_interval
                        print(f"Episode: {episode}, Reward: {avg_R:.2f}, Avg. syntax {avg_syntax_r:.3f}")
                    avg_syntax_r = 0

        return loss_history, reward_history


    def test(self,train=False):

        self.train(train=train)

        return loss_history, reward_history




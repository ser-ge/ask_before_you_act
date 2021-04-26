import numpy as np
from collections import namedtuple

import torch
import wandb

Transition = namedtuple('Transition', ['state', 'answer', 'word_lstm_hidden',
                                       'action', 'reward', 'reward_qa', 'next_state', 'log_prob_act',
                                       'log_prob_qa', 'entropy_act', 'entropy_qa', 'done'])


def train(env, agent, logger, exploration=True, n_episodes=1000,
          log_interval=50, verbose=False):
    episode = 0
    episode_loss = 0

    episode_reward = []
    episode_qa_reward = []
    loss_history = []
    reward_history = []

    state = env.reset()['image']  # Discard other info
    step = 0

    avg_syntax_r = 0

    qa_pairs = []

    while episode < n_episodes:
        # Ask - TODO pass qa_history
        question, word_lstm_hidden, log_prob_qa, entropy_qa = agent.ask(state)

        # Answer
        answer, reward_qa = env.answer(question)
        qa_pairs.append([question, str(answer), reward_qa])

        answer = answer.decode()

        avg_syntax_r += 1 / log_interval * (reward_qa - avg_syntax_r)  # TODO double check if this is correct
        # answer = [0, 0]
        # word_lstm_hidden = torch.zeros_like(word_lstm_hidden).detach()

        # Act
        action, log_prob_act, entropy_act = agent.act(state, answer, word_lstm_hidden)

        # Step
        next_state, reward, done, _ = env.step(action)

        # if reward > 0:
        #     print("Goal reached")

        next_state = next_state['image']  # Discard other info
        # Store
        if exploration:
            t = Transition(state, answer, word_lstm_hidden, action, reward,
                           reward_qa, next_state, log_prob_act.item(), log_prob_qa,
                           entropy_act.item(), entropy_qa, done)
            agent.store(t)

        # Advance
        state = next_state
        step += 1

        # Logging
        episode_reward.append(reward)
        episode_qa_reward.append(reward_qa)

        if done:
            # Update
            if exploration:
                episode_loss = agent.update()

            state = env.reset()['image']  # Discard other info
            step = 0

            loss_history.append(episode_loss)
            reward_history.append(sum(episode_reward))

            logger.log(
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

            if episode % log_interval == 0:
                if verbose:
                    avg_R = np.sum(reward_history[-log_interval:]) / log_interval
                    print(f"Episode: {episode}, Reward: {avg_R:.2f}, Avg. syntax {avg_syntax_r:.3f}")
                avg_syntax_r = 0

    return loss_history, reward_history

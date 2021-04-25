import numpy as np
from collections import namedtuple

import torch

Transition = namedtuple('Transition', ['state', 'answer', 'word_lstm_hidden',
                                       'action', 'reward', 'reward_qa', 'next_state', 'log_prob_act',
                                       'log_prob_qa', 'entropy_act', 'entropy_qa', 'done'])

def train(env, agent, exploration=True, n_episodes=1000,
             log_interval=50, verbose=False, ID=False):
    episode = 0
    episode_loss = 0

    episode_reward = []
    loss_history = []
    reward_history = []

    state = env.reset()['image']  # Discard other info
    step = 0

    avg_syntax_r = 0

    while episode < n_episodes:
        # Ask - TODO pass qa_history
        question, word_lstm_hidden, log_prob_qa, entropy_qa = agent.ask(state)

        # Answer
        answer, reward_qa = env.answer(question)
        avg_syntax_r += 1/log_interval * (reward_qa - avg_syntax_r)
        answer = [0, 0]
        word_lstm_hidden = torch.zeros_like(word_lstm_hidden)

        # Act
        action, log_prob_act, entropy_act = agent.act(state, answer, word_lstm_hidden)

        # if sum(answer) == 2 or sum(answer) == 0:
        #     print(question)
        #     print(answer)
        #     print(reward_qa)
        #     env.render()
        #     sleep(0.001)

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

        if done:
            # Update
            if exploration:
                episode_loss = agent.update()

            state = env.reset()['image']  # Discard other info
            step = 0

            loss_history.append(episode_loss)
            reward_history.append(sum(episode_reward))
            episode_reward = []

            episode += 1

            if episode % log_interval == 0:
                if verbose:
                    avg_R = np.sum(reward_history[-log_interval:]) / log_interval
                    print(f"Episode: {episode}, Reward: {avg_R:.2f}, Avg. syntax {avg_syntax_r:.3f}")

    return loss_history, reward_history
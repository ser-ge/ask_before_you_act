#!/usr/bin/env python

import numpy as np


def GAEtrain(env, agent, exploration=True, n_episodes=1000,
             log_interval=50, verbose=False, ID=False):
    episode = 0
    episode_loss = 0

    episode_reward = []
    loss_history = []
    reward_history = []

    state = env.reset()['image']  # Discard other info
    step = 0

    while episode < n_episodes:
        # Act
        a, log_prob, entropy = agent.act(state, exploration)

        # Step
        next_state, r, done, _ = env.step(a)
        next_state = next_state['image']  # Discard other info
        # Store
        if exploration:
            agent.store((state, a, r, next_state,
                         log_prob.item(), entropy.item(), done))

        # Advance
        state = next_state
        step += 1

        # Logging
        episode_reward.append(r)

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

            if (episode) % log_interval == 0:
                if verbose:
                    avg_R = np.sum(reward_history[-log_interval:])/log_interval
                    print(f"Episode: {episode}, Reward: {avg_R:.1f}")

    return loss_history, reward_history

#!/usr/bin/env python3
from time import sleep

import numpy as np

# TODO - Unify trainer for PPOAgentMem and PPOAgent


def GAEtrain(env, agent, logger, n_episodes=1000,
             log_interval=50, verbose=False):
    episode = 0

    episode_reward = []
    loss_history = []
    reward_history = []

    state = env.reset()['image']  # Discard other info
    step = 0

    while episode < n_episodes:
        # Act
        a, log_prob, entropy = agent.act(state)

        # Step
        next_state, r, done, _ = env.step(a)
        next_state = next_state['image']  # Discard other info
        # env.render()

        # Store
        agent.store((state, a, r, next_state,
                     log_prob.item(), entropy.item(), done))

        # Advance
        state = next_state
        step += 1

        # Logging
        episode_reward.append(r)

        if done:
            # Update
            episode_loss = agent.update()

            state = env.reset()['image']  # Discard other info
            step = 0

            loss_history.append(episode_loss)
            reward_history.append(sum(episode_reward))

            logger.log(
                {
                    "eps_reward": sum(episode_reward),
                    "loss": episode_loss
                }
            )

            episode_reward = []
            episode += 1

            if episode % log_interval == 0:
                if verbose:
                    avg_R = np.sum(reward_history[-log_interval:])/log_interval
                    print(f"Episode: {episode}, Reward: {avg_R:.1f}")

    return loss_history, reward_history

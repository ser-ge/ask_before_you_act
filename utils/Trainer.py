import torch
from time import sleep

from utils.language import adjective, noun, verb, direction, state
import numpy as np

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
        question, hx, log_probs_qa, entropy_qa = agent.ask(state)

        # Answer
        ans, q_reward = env.answer(question)

        # Act
        a, log_prob, entropy = agent.act(state, ans, hx)

        if episode % (log_interval*2) == 0:
            print(question)
            print(ans)
            print(q_reward)
        #     env.render()
            # sleep(0.001)

            # Step
        next_state, r, done, _ = env.step(a)

        if r > 0:
            print("Goal reached")

        r += q_reward # TODO store
        next_state = next_state['image']  # Discard other info
        # Store
        if exploration:
            agent.store((state, ans, hx, a, r, next_state,
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
                    avg_R = np.sum(reward_history[-log_interval:]) / log_interval
                    print(f"Episode: {episode}, Reward: {avg_R:.2f}, Avg. syntax {avg_syntax_r:.3f}")

    return loss_history, reward_history
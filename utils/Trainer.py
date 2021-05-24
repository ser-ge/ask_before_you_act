import time

import numpy as np
from collections import namedtuple
from oracle.oracle import Answer
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
        "log_prob_act",
        "log_prob_qa",
        "entropy_act",
        "entropy_qa",
        "done",
        "hidden_hist_mem",
        "cell_hist_mem",
    ],
)

def train_test(env, agent, cfg, logger, n_episodes=1000,
          log_interval=50, train=True, verbose=True):
    episode = 0

    episode_reward = []
    episode_qa_reward = []
    loss_history = []
    reward_history = []

    state = env.reset()['image']  # Discard other info
    step = 0

    avg_syntax_r = 0

    qa_pairs = []

    # Initialize random memory
    hist_mem = agent.init_memory()
    last_time = time.time()

    while episode < n_episodes:
        # Ask before you act
        if cfg.baseline:
            action, log_prob_act, entropy_act = agent.act(state, hist_mem[0])
            answer, reward_qa, entropy_qa = (1,0,1)
            log_prob_qa = 6*[torch.Tensor([1])]
            hidden_q = torch.ones(128)

        else:
            # Ask
            question, hidden_q, log_prob_qa, entropy_qa = agent.ask(state, hist_mem[0])
            answer, reward_qa = env.answer(question)

            # Logging
            episode_qa_reward.append(reward_qa)
            qa_pairs.append([question, str(answer), reward_qa])  # Storing

            # Answer
            answer = answer.encode()  # For passing vector to agent
            avg_syntax_r += 1 / log_interval * (reward_qa - avg_syntax_r)

            # Act
            action, log_prob_act, entropy_act = agent.act(state, answer, hidden_q, hist_mem[0])

        # Remember
        if cfg.use_mem:  # need to make this work for baseline also
            if cfg.baseline:
                next_hist_mem = agent.remember(state, action, hist_mem)
            else:
                next_hist_mem = agent.remember(state, action, answer, hidden_q, hist_mem)
        else:
            next_hist_mem = agent.init_memory()

        # Step
        next_state, reward, done, _ = env.step(action)
        next_state = next_state['image']  # Discard other info

        # Store
        t = Transition(state, answer, hidden_q, action, reward, reward_qa,
                       log_prob_act.item(), log_prob_qa, entropy_act.item(), entropy_qa, done,
                       hist_mem[0], hist_mem[1])

        agent.store(t)

        # Advance
        state = next_state
        hist_mem = next_hist_mem  # Random hist_mem if nto using memory
        step += 1

        # Logging
        episode_reward.append(reward)

        if done:
            # Update
            if train:
                episode_loss, losses_tuple = agent.update()
                loss_history.append(episode_loss)

            # Reset episode
            state = env.reset()['image']  # Discard other info
            hist_mem = agent.init_memory()  # Initialize memory
            step = 0

            reward_history.append(sum(episode_reward))
            if cfg.wandb:
                log_cases(logger, cfg, episode, episode_loss, losses_tuple, episode_qa_reward,
                          episode_reward, qa_pairs, reward_history, train)


            episode_reward = []
            episode_qa_reward = []
            qa_pairs = []

            episode += 1

            if episode % log_interval == 0:
                current_time = time.time()
                if verbose:
                    avg_R = np.mean(reward_history[-log_interval:])
                    print(f"Episode: {episode}, Reward: {avg_R:.2f}, Avg. syntax {avg_syntax_r:.3f}, "
                          f"EPS: {log_interval / (current_time - last_time):.1f} ")
                avg_syntax_r = 0
                last_time = current_time

    return reward_history


def log_cases(logger, cfg, episode, episode_loss, losses_tuple, episode_qa_reward,
              episode_reward, qa_pairs, reward_history, train):
    if train:
        L_clip, L_value, L_entropy, L_policy_qa, L_entropy_qa = losses_tuple
        if cfg.baseline:
            logger.log(
                {
                    "train/eps_reward": sum(episode_reward),
                    "train/total_loss": episode_loss,
                    "train/L_clip": abs(L_clip),
                    "train/L_value": abs(L_value),
                    "train/L_entropy": abs(L_entropy),
                    "train/avg_reward_episodes": sum(reward_history) / len(reward_history)
                }
            )
        else:
            logger.log(
                {
                    "train/eps_reward": sum(episode_reward),
                    "train/avg_reward_qa": sum(episode_qa_reward) / len(episode_qa_reward),
                    "train/loss": episode_loss,
                    "train/L_clip": abs(L_clip),
                    "train/L_value": abs(L_value),
                    "train/L_entropy": abs(L_entropy),
                    "train/L_policy_qa": abs(L_policy_qa),
                    "train/L_entropy_qa": abs(L_entropy_qa),
                    "train/avg_reward_episodes": sum(reward_history) / len(reward_history)
                }
            )
            if episode % cfg.train_log_interval == 0:
                logger.log({"train/questions": wandb.Table(data=qa_pairs, columns=["Question", "Answer", "Reward"])})

        if cfg.baseline:
            logger.log(
                {
                    "test/eps_reward": sum(episode_reward),
                    "test/avg_reward_episodes": sum(reward_history) / len(reward_history)
                }
            )
        else:
            logger.log(
                {
                    # "test/questions": wandb.Table(data=qa_pairs, columns=["Question", "Answer", "Reward"]),
                    "test/eps_reward": sum(episode_reward),
                    "test/avg_reward_qa": sum(episode_qa_reward) / len(episode_qa_reward),
                    "test/avg_reward_episodes": sum(reward_history) / len(reward_history)
                }
            )

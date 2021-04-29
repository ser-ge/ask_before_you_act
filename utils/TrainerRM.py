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

    while episode < n_episodes:
        # Ask before you act
        if cfg.baseline:
            action, log_prob_act, entropy_act = agent.act(state, hist_mem[0])
            answer = 1
            reward_qa = 0
            entropy_qa = 1
            log_prob_qa = 6*[torch.Tensor([1])]
            hidden_q = torch.ones(128)
            # question = 'wherefore art thou Romeo?'

        else:
            # Ask
            question, hidden_q, log_prob_qa, entropy_qa = agent.ask(state, hist_mem[0])
            answer, reward_qa = env.answer(question)

            # Logging
            episode_qa_reward.append(reward_qa)
            qa_pairs.append([question, str(answer), reward_qa])  # Storing

            # Answer
            answer = answer.decode()  # For passing vector to agent
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
        t = Transition(state, answer, hidden_q, action, reward, reward_qa, next_state,
                       log_prob_act.item(),log_prob_qa, entropy_act.item(), entropy_qa, done,
                       hist_mem[0], hist_mem[1], next_hist_mem[0], next_hist_mem[1])

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
                episode_loss = agent.update()
                loss_history.append(episode_loss)

            # Reset episode
            state = env.reset()['image']  # Discard other info
            hist_mem = agent.init_memory()  # Initialize memory
            step = 0

            reward_history.append(sum(episode_reward))

            log_cases(logger, cfg, episode_loss, episode_qa_reward,
                      episode_reward, qa_pairs, reward_history, train)

            episode_reward = []
            episode_qa_reward = []
            qa_pairs = []

            episode += 1

            if episode % log_interval == 0:
                if verbose:
                    avg_R = np.sum(reward_history[-log_interval:]) / log_interval
                    print(f"Episode: {episode}, Reward: {avg_R:.2f}, Avg. syntax {avg_syntax_r:.3f}")
                avg_syntax_r = 0

    return reward_history


def log_cases(logger, cfg, episode_loss, episode_qa_reward, episode_reward, qa_pairs, reward_history, train):
    if train:
        if cfg.baseline:
            logger.log(
                {
                    "train/eps_reward": sum(episode_reward),
                    "train/loss": episode_loss,
                    "train/avg_reward_episodes": sum(reward_history) / len(reward_history)
                }
            )
        else:
            logger.log(
                {
                    "train/questions": wandb.Table(data=qa_pairs, columns=["Question", "Answer", "Reward"]),
                    "train/eps_reward": sum(episode_reward),
                    "train/avg_reward_qa": sum(episode_qa_reward) / len(episode_qa_reward),
                    "train/loss": episode_loss,
                    "train/avg_reward_episodes": sum(reward_history) / len(reward_history)
                }
            )
    else:
        if cfg.baseline:
            logger.log(
                {
                    "test/eps_reward": sum(episode_reward),
                    "test/loss": episode_loss,
                    "test/avg_reward_episodes": sum(reward_history) / len(reward_history)
                }
            )
        else:
            logger.log(
                {
                    "test/questions": wandb.Table(data=qa_pairs, columns=["Question", "Answer", "Reward"]),
                    "test/eps_reward": sum(episode_reward),
                    "test/avg_reward_qa": sum(episode_qa_reward) / len(episode_qa_reward),
                    "test/loss": episode_loss,
                    "test/avg_reward_episodes": sum(reward_history) / len(reward_history)
                }
            )

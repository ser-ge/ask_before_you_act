import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

# Import data

baseline_train_ans_true = np.load("./data/baseline_train_ans_true.npy")[:, 1:]
film_train_ans_true = np.load("./data/film_train_ans_true.npy")[:, 2:]  # TODO - nan in first run!
main_train_ans_true = np.load("./data/main_train_ans_true.npy")[:, 1:]

baseline_test_ans_true = np.load("./data/baseline_test_ans_true.npy")[:, 1:]
film_test_ans_true = np.load("./data/film_test_ans_true.npy")[:, 1:]
main_test_ans_true = np.load("./data/main_test_ans_true.npy")[:, 1:]

film_train_ans_random = np.load("./data/film_train_ans_random.npy")[:, 1:]
film_test_ans_random = np.load("./data/film_test_ans_random.npy")[:, 1:]

data = [baseline_train_ans_true, film_train_ans_true, main_train_ans_true, baseline_test_ans_true,
        film_test_ans_true, main_test_ans_true, film_train_ans_random, film_test_ans_random]

# Some stats
means = [item.mean(1) for item in data]
stds = [item.std(1) for item in data]

#
# fig, axs = plt.subplots(2, 1)
# episodes = np.linspace(0, 4999, num=5000)
# axs[0].plot(film_train_ans_true.mean(1))
# # axs[0].fill_between(episodes,
# #                     film_train_ans_true.mean(1) - film_train_ans_true.std(1),
# #                     film_train_ans_true.mean(1) + film_train_ans_true.std(1), alpha=0.3)
# axs[0].set_xlabel("Episodes")
# axs[0].set_ylabel("Training reward")
# plt.show()

# Averages
window = 100
means_smoothed = [(window - 1) * [0]
                           + pd.Series(item).rolling(window).mean().to_list()[window - 1:] for item in means]
stds_smoothed = [(window - 1) * [0]
                          + pd.Series(item).rolling(window).mean().to_list()[window - 1:] for item in stds]


# Plot generalisation
fig, axs = plt.subplots(2, 1)
labels = ["baseline_train_ans_true", "film_train_ans_true", "main_train_ans_true"]
for mean, std, label in zip(means_smoothed[0:3], stds_smoothed[0:3], labels):
    mean = np.array(mean)[:5000]
    std = np.array(std)[:5000]
    episodes = np.linspace(0, mean.shape[0] - 1, num=mean.shape[0])
    axs[0].plot(episodes, mean, label=label)
    axs[0].fill_between(episodes, mean - std, mean + std, alpha=0.3)
axs[0].set_xlabel("Episodes")
axs[0].set_ylabel("Training reward")
axs[0].set_title("Generalisation")
axs[0].legend()

labels = ["baseline_test_ans_true", "film_test_ans_true", "main_test_ans_true"]
for mean, std, label in zip(means_smoothed[3:6], stds_smoothed[3:6], labels):
    mean = np.array(mean)[:5000]
    std = np.array(std)[:5000]
    episodes = np.linspace(0, mean.shape[0] - 1, num=mean.shape[0])
    axs[1].plot(episodes, mean, label=label)
    axs[1].fill_between(episodes, mean - std, mean + std, alpha=0.3)
axs[1].set_xlabel("Episodes")
axs[1].set_ylabel("Test reward")
axs[1].legend()
plt.tight_layout()
plt.show()

t_now = str(time.asctime()).replace(" ", "_")
fig.savefig("./figures/generalisation_" + t_now + ".png")

#####################################################################################

# data = [baseline_train_ans_true, film_train_ans_true, main_train_ans_true, baseline_test_ans_true,
#         film_test_ans_true, main_test_ans_true, film_train_ans_random, film_test_ans_random]

# Plot noisy 1
fig, axs = plt.subplots(2, 1)
labels = ["baseline_train_ans_true", "film_train_ans_true", "film_train_ans_random"]
means = [means_smoothed[0], means_smoothed[1], means_smoothed[-2]]
stds = [stds_smoothed[0], stds_smoothed[1], stds_smoothed[-2]]
for mean, std, label in zip(means, stds, labels):
    mean = np.array(mean)[:5000]
    std = np.array(std)[:5000]
    episodes = np.linspace(0, mean.shape[0] - 1, num=mean.shape[0])
    axs[0].plot(episodes, mean, label=label)
    axs[0].fill_between(episodes, mean - std, mean + std, alpha=0.3)
axs[0].set_title("Noisy Oracle")
axs[0].set_xlabel("Episodes")
axs[0].set_ylabel("Training reward")
axs[0].legend()

labels = ["baseline_test_ans_true", "film_test_ans_true", "film_test_ans_random"]
means = [means_smoothed[3], means_smoothed[4], means_smoothed[-1]]
stds = [stds_smoothed[3], stds_smoothed[4], stds_smoothed[-1]]
for mean, std, label in zip(means, stds, labels):
    mean = np.array(mean)[:5000]
    std = np.array(std)[:5000]
    episodes = np.linspace(0, mean.shape[0] - 1, num=mean.shape[0])
    axs[1].plot(episodes, mean, label=label)
    axs[1].fill_between(episodes, mean - std, mean + std, alpha=0.3)
axs[1].set_xlabel("Episodes")
axs[1].set_ylabel("Test reward")
axs[1].legend()
plt.tight_layout()
plt.show()

t_now = str(time.asctime()).replace(" ", "_")
fig.savefig("./figures/noisy_oracle_" + t_now + ".png")

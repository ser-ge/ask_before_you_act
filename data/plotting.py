import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time


def plotter(plots_list, window=200):
    for plot in plots_list:
        dict = {}
        for subplot in plot:
            for experiment in subplot:
                if experiment == "film_train_ans_true":
                    dict[experiment] = np.load("./data/" + experiment + ".npy")[:, 2:]
                else:
                    dict[experiment] = np.load("./data/" + experiment + ".npy")[:, 1:]

        # Some stats
        means_train = [dict[experiment].mean(1) for experiment in plot[0]]
        stds_train = [dict[experiment].std(1) for experiment in plot[0]] #np.sqrt(dict[experiment].shape[0])

        means_test = [dict[experiment].mean(1) for experiment in plot[1]]
        stds_test = [dict[experiment].std(1) for experiment in plot[1]]

        # Averages
        means_train_smoothed = [(window - 1) * [0]
                                   + pd.Series(item).rolling(window).mean().to_list()[window - 1:] for item in means_train]
        stds_train_smoothed = [(window - 1) * [0]
                                  + pd.Series(item).rolling(window).mean().to_list()[window - 1:] for item in stds_train]

        means_test_smoothed = [(window - 1) * [0]
                                   + pd.Series(item).rolling(window).mean().to_list()[window - 1:] for item in means_test]
        stds_test_smoothed = [(window - 1) * [0]
                                  + pd.Series(item).rolling(window).mean().to_list()[window - 1:] for item in stds_test]

        # Plot generalisation
        plt.style.use("seaborn")
        fig, axs = plt.subplots(2, 1, figsize=(4.5, 6.75))
        labels = ["Baseline", "Main", "FiLM"]
        for mean, std, label in zip(means_train_smoothed, stds_train_smoothed, labels):
            mean = np.array(mean)[:5000]
            std = np.array(std)[:5000]
            episodes = np.linspace(0, mean.shape[0] - 1, num=mean.shape[0])
            axs[0].plot(episodes, mean, label=label)
            # axs[0].fill_between(episodes, mean - std, mean + std, alpha=0.3)
        # axs[0].set_xlabel("Episodes")
        axs[0].set_ylabel("Training reward")
        axs[0].legend(loc="upper left")

        labels = ["Baseline", "Main", "Main NoEmbed", "Main Rand", "FiLM", "FiLM Rand"]
        styles = ["#1f77b4", "#2ca02c", "#2ca02c", "#2ca02c", "#d62728", "#d62728"]
        alphas = [0.8, 1,  0.65, 0.35, 1, 0.35]
        for mean, std, label, sty, a in zip(means_test_smoothed, stds_test_smoothed, labels, styles, alphas):
            mean = np.array(mean)[:5000]
            std = np.array(std)[:5000]
            episodes = np.linspace(0, mean.shape[0] - 1, num=mean.shape[0])
            axs[1].plot(episodes, mean, sty, label=label, alpha=a, linewidth=1.25)
            # axs[1].fill_between(episodes, mean - std, mean + std, alpha=0.3)
        axs[1].set_xlabel("Episodes")
        axs[1].set_ylabel("Test reward")
        axs[1].legend(loc="upper left")
        plt.tight_layout()
        plt.show()

        time.sleep(0.5)
        t_now = str(time.asctime()).replace(" ", "_")
        fig.savefig("./figures/plot_" + t_now + ".png")


if __name__ == '__main__':
    # plot1_train = ["baseline_train_ans_true", "main_em_non_random_train"]
    # plot1_test = ["baseline_test_ans_true", "main_em_non_random_test"]
    # plot1 = [plot1_train, plot1_test]

    # plot2_train = ["baseline_train_ans_true", "film_train_ans_true", "film_train_ans_random"]
    # plot2_test = ["baseline_test_ans_true", "film_test_ans_true", "film_test_ans_random"]
    # plot2 = [plot2_train, plot2_test]
    #
    plot3_train = ["baseline_train_ans_true", "main_em_non_random_train", "film_train_ans_true"]
    plot3_test = ["baseline_test_ans_true", "main_em_non_random_test", "main_test_ans_true",
                  "main_em_random_test", "film_test_ans_true", "film_test_ans_random"]
    plot3 = [plot3_train, plot3_test]

    plots_list = [plot3]

    plotter(plots_list)



    # # Import data
    #
    # baseline_train_ans_true = np.load("./data/baseline_train_ans_true.npy")[:, 1:]
    # film_train_ans_true = np.delete(np.load("./data/film_train_ans_true.npy"), 1, 1)[:, 1:]  # TODO - nan in first run!
    # main_train_ans_true = np.load("./data/main_train_ans_true.npy")[:, 1:]
    #
    # baseline_test_ans_true = np.load("./data/baseline_test_ans_true.npy")[:, 1:]
    # film_test_ans_true = np.load("./data/film_test_ans_true.npy")[:, 1:]
    # main_test_ans_true = np.load("./data/main_test_ans_true.npy")[:, 1:]
    #
    # film_train_ans_random = np.load("./data/film_train_ans_random.npy")[:, 1:]
    # film_test_ans_random = np.load("./data/film_test_ans_random.npy")[:, 1:]
    #
    #
    # data = [baseline_train_ans_true, film_train_ans_true, main_train_ans_true, baseline_test_ans_true,
    #         film_test_ans_true, main_test_ans_true, film_train_ans_random, film_test_ans_random]
    #
    # # Some stats
    # means = [item.mean(1) for item in data]
    # stds = [item.std(1) for item in data]
    #
    # # Averages
    # window = 100
    # means_smoothed = [(window - 1) * [0]
    #                            + pd.Series(item).rolling(window).mean().to_list()[window - 1:] for item in means]
    # stds_smoothed = [(window - 1) * [0]
    #                           + pd.Series(item).rolling(window).mean().to_list()[window - 1:] for item in stds]
    #
    #
    # # Plot generalisation
    # fig, axs = plt.subplots(2, 1)
    # labels = ["Baseline", "Main"]
    # for mean, std, label in zip(means_smoothed[0], stds_smoothed[2], labels):
    #     mean = np.array(mean)[:5000]
    #     std = np.array(std)[:5000]
    #     episodes = np.linspace(0, mean.shape[0] - 1, num=mean.shape[0])
    #     axs[0].plot(episodes, mean, label=label)
    #     axs[0].fill_between(episodes, mean - std, mean + std, alpha=0.3)
    # axs[0].set_xlabel("Episodes")
    # axs[0].set_ylabel("Training reward")
    # axs[0].set_title("Generalisation")
    # axs[0].legend()
    #
    # labels = ["Baseline", "Main"]
    # for mean, std, label in zip(means_smoothed[3], stds_smoothed[5], labels):
    #     mean = np.array(mean)[:5000]
    #     std = np.array(std)[:5000]
    #     episodes = np.linspace(0, mean.shape[0] - 1, num=mean.shape[0])
    #     axs[1].plot(episodes, mean, label=label)
    #     axs[1].fill_between(episodes, mean - std, mean + std, alpha=0.3)
    # axs[1].set_xlabel("Episodes")
    # axs[1].set_ylabel("Test reward")
    # axs[1].legend()
    # plt.tight_layout()
    # plt.show()
    #
    # t_now = str(time.asctime()).replace(" ", "_")
    # fig.savefig("./figures/generalisation_" + t_now + ".png")
#
# #####################################################################################
#
# # data = [baseline_train_ans_true, film_train_ans_true, main_train_ans_true, baseline_test_ans_true,
# #         film_test_ans_true, main_test_ans_true, film_train_ans_random, film_test_ans_random]
#
# # Plot noisy 1
# fig, axs = plt.subplots(2, 1)
# labels = ["baseline_train_ans_true", "film_train_ans_true", "film_train_ans_random"]
# means = [means_smoothed[0], means_smoothed[1], means_smoothed[-2]]
# stds = [stds_smoothed[0], stds_smoothed[1], stds_smoothed[-2]]
# for mean, std, label in zip(means, stds, labels):
#     mean = np.array(mean)[:5000]
#     std = np.array(std)[:5000]
#     episodes = np.linspace(0, mean.shape[0] - 1, num=mean.shape[0])
#     axs[0].plot(episodes, mean, label=label)
#     axs[0].fill_between(episodes, mean - std, mean + std, alpha=0.3)
# axs[0].set_title("Noisy Oracle")
# axs[0].set_xlabel("Episodes")
# axs[0].set_ylabel("Training reward")
# axs[0].legend()
#
# labels = ["baseline_test_ans_true", "film_test_ans_true", "film_test_ans_random"]
# means = [means_smoothed[3], means_smoothed[4], means_smoothed[-1]]
# stds = [stds_smoothed[3], stds_smoothed[4], stds_smoothed[-1]]
# for mean, std, label in zip(means, stds, labels):
#     mean = np.array(mean)[:5000]
#     std = np.array(std)[:5000]
#     episodes = np.linspace(0, mean.shape[0] - 1, num=mean.shape[0])
#     axs[1].plot(episodes, mean, label=label)
#     axs[1].fill_between(episodes, mean - std, mean + std, alpha=0.3)
# axs[1].set_xlabel("Episodes")
# axs[1].set_ylabel("Test reward")
# axs[1].legend()
# plt.tight_layout()
# plt.show()
#
# t_now = str(time.asctime()).replace(" ", "_")
# fig.savefig("./figures/noisy_oracle_" + t_now + ".png")

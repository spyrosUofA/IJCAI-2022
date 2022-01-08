import numpy as np
import argparse
import matplotlib.pyplot as plt
from shutil import copyfile

# 0: ID
# 1: Reward
# 2: Score
# 3: Nb Games Played
# 4: Times


parser = argparse.ArgumentParser()
parser.add_argument('-config', action='store', dest='config', default="E010_D010/64x64")
parser.add_argument('-hparam', action='store', dest='hparam', default="sa_cpus-1_n-50_c-None")
parser.add_argument('-nb_oracles', action='store', dest='nb_oracles', default=15)
parser.add_argument('-nb_resets', action='store', dest='nb_resets', default=30)
parameters = parser.parse_args()

NB_ORACLES = int(parameters.nb_oracles)
NB_RESETS = int(parameters.nb_resets)
RUN_TIME = 7200.0
WORST_SCORE = -1000


def data_oracle(oracle, nb_seeds):

    rewards_Y = []
    for i in range(nb_seeds):
        if oracle == "BaseDSL":
            name = str(oracle) + '/' + str(i + 1)
            rewards_Y.append(np.load("./Oracle/" + name + "/BaseTreeRewards.npy").tolist())
        else:
            name = str(oracle) + '/' + str(i + 1)
            rewards_Y.append(np.load("./Oracle/" + name + "/AugTreeRewards.npy").tolist())

    depths_X = [1, 2, 3]
    mean_Y = np.mean(rewards_Y, axis=0)
    std_Y = np.std(rewards_Y, axis=0) #* (nb_seeds ** -0.5)

    print(mean_Y, std_Y)
    return depths_X, mean_Y, std_Y


def view_oracle(oracle, nb_seeds):

    rewards_Y = []
    for i in range(nb_seeds):
        if oracle == "BaseDSL":
            name = str(oracle) + '/' + str(i + 1)
            rewards_Y.append(np.load("./Oracle/" + name + "/BaseTreePrograms.npy").tolist())
        else:
            name = str(oracle) + '/' + str(i + 1)
            rewards_Y.append(np.load("./Oracle/" + name + "/AugTreePrograms.npy").tolist())

    depths_X = [1, 2, 3]
    mean_Y = np.mean(rewards_Y, axis=0)
    std_Y = np.std(rewards_Y, axis=0) #* (nb_seeds ** -0.5)

    print(mean_Y, std_Y)
    return depths_X, mean_Y, std_Y


def plot_all(configs, seeds):

    for count, config in enumerate(configs):
        depths_X, mean_Y, std_Y = data_oracle(config, seeds)
        plt.plot(depths_X, mean_Y)
        plt.fill_between(depths_X, mean_Y - std_Y, mean_Y + std_Y, alpha=0.2)

    plt.xlabel('Tree Depth')
    plt.ylabel('Reward (100 episodes)')
    plt.title('LunarLander-v2')
    plt.ylim([0, 510])
    plt.hlines(485., 1, 3, colors='red', linestyles='dashed')
    plt.legend(configs, loc='lower right')
    plt.pause(5)
    plt.savefig('TreePlot.png', dpi=1080, bbox_inches="tight")


plot_all(["4x0", "32x0", "64x64", "256x256", "BaseDSL"], 15)



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


def plot_one(oracle_nb):
    episodes_X = []
    times_X = []
    rewards_Y = []
    scores_Y = []
    
    best_score = WORST_SCORE
    
    for r in range(1, NB_RESETS + 1, 1):
        file = "../logs/" + parameters.config + '/' + str(oracle_nb) + '/' + parameters.hparam + "_run-" + str(r)

        with open(file) as f:
            output = np.array([line.strip().split(', ') for line in f], dtype=float)

        current_score = max(output[:, 1])
        if current_score > best_score:
            best_score = current_score
            best_policy = str(r)

        # Read  games, times, rewards, and scores
        rewards_Y.extend(output[:, 1])
        scores_Y.extend(output[:, 2])
        episodes_X.extend(output[:, 3])

        # Add previous run times
        times = output[:, 4] + (r-1) * RUN_TIME
        times_X.extend(times) 

    # Accumulate maximum reward and score
    rewards_Y = np.maximum.accumulate(rewards_Y)
    scores_Y = np.maximum.accumulate(scores_Y)

    # save best policy
    copy_from = parameters.config + '/' + str(oracle_nb) + '/' + parameters.hparam + "_run-" + best_policy
    save_to = parameters.config + '/' + str(oracle_nb) + '/' + parameters.hparam + "_run-BEST"

    copyfile("../logs/" + copy_from, "../logs/" + save_to)
    copyfile("../programs/" + copy_from, "../programs/" + save_to)
    copyfile("../binary_programs/" + copy_from + ".pkl", "../binary_programs/" + save_to + ".pkl")

    return [times_X, rewards_Y] # ...


def plot_all():
    
    for i in range(1, NB_ORACLES+1):
        current_oracle = plot_one(i)
        plt.plot(current_oracle[0], current_oracle[1])
    
    plt.title(parameters.config)
    plt.xlabel('Sequential Runtime (hrs)')
    plt.ylabel('Reward')
    plt.ylim([-200, 250])
    plt.legend(range(1, NB_ORACLES+1), title="Oracle", loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('./plots/' + parameters.config + ".png", dpi=1080, bbox_inches="tight")
    plt.pause(5)
    plt.close()


def get_line():

    results = []
    extended_times = []
    extended_scores = []

    # Gather all Times and Scores
    for i in range(1, NB_ORACLES + 1):
        current_oracle = plot_one(i)
        results.append(current_oracle)
        extended_times.extend(current_oracle[0])
    extended_times.sort()

    # Extent score vectors to be consistent with extended times
    for i in range(1, NB_ORACLES + 1):
        # Current Run i
        times_i = results[i-1][0]
        scores_i = results[i-1][1]
        # Vector of length all_times
        extended_scores_i = [scores_i[0]] * len(extended_times)
        # Indices of current times in all_times vector
        indexes_i = [i in times_i for i in extended_times]
        indexes_i = np.where(indexes_i)[0]
        # Fill up an extended vector with the
        for x, y in zip(indexes_i, scores_i):
            extended_scores_i[x] = y
        #extended_scores_i[indexes_i] = scores_i
        extended_scores_i = np.maximum.accumulate(extended_scores_i)
        # Update results
        extended_scores.append(extended_scores_i)

    # Take averages
    score_avg = np.mean(extended_scores, axis=0)
    score_std = np.std(extended_scores, axis=0) / (NB_ORACLES ** 0.5)
    
    np.save(file='./plot_data/' + parameters.config + '_time.npy', arr=extended_times)
    np.save(file='./plot_data/' + parameters.config + '_avg.npy', arr=score_avg)
    np.save(file='./plot_data/' + parameters.config + '_std.npy', arr=score_std)

    # for i in range(1, 16):
    #     print(extended_scores[i-1][-5:])

    plt.plot(extended_times, score_avg)
    plt.fill_between(extended_times, score_avg - score_std, score_avg + score_std, alpha=0.2)
    plt.xlabel('Sequential Runtime (hrs)')
    plt.ylabel('Reward')
    plt.title(parameters.config)
    plt.ylim([-200, 250])
    plt.savefig('./plots/' + parameters.config + "_Avg.png", dpi=1080, bbox_inches="tight")
    plt.pause(10)

    return score_avg, score_std

plot_all()
get_line()

import numpy as np
import argparse
import matplotlib.pyplot as plt
from shutil import copyfile


def avg_final_rew(oracle, approach, nb_seeds, depth):
    rew_final = []
    for i in range(nb_seeds):
        load_from = './Oracle/' + str(oracle) + '/' + str(i+1) + '/' + approach + "/Rew_" + str(depth) + '.npy'
        rew_final.append(np.load(load_from).tolist())
    mean_Y = np.mean(rew_final, axis=0)
    std_Y = np.std(rew_final, axis=0)  # * (nb_seeds ** -0.5)
    print(oracle, approach, ":", mean_Y, std_Y)
    return mean_Y, std_Y


def avg_rew_vs_time(oracle, approach, nb_seeds, depth):

    results = []
    extended_times = []
    extended_scores = []
    times = []
    rewards = []
    for i in range(nb_seeds):
        load_from = './Oracle/' + str(oracle) + '/' + str(i+1) + '/' + approach + "/TimeVsReward_" + str(depth) + '.npy'
        times_and_rewards = np.load(load_from)
        t_i = [item[0] for item in times_and_rewards]  #times_and_rewards[:][1]
        r_i = [item[1] for item in times_and_rewards] #times_and_rewards[:][0]

        rewards.append(r_i)
        times.append(t_i)

        extended_times.extend(t_i)
        results.append(times_and_rewards)
    extended_times.sort()

    worst_score = 0 # min(rewards)
    print(worst_score)

    print(results)
    print(extended_times)

    # Extent score vectors to be consistent with extended times
    for i in range(1, nb_seeds + 1):
        # Current Run i
        scores_i = rewards[i-1] # results[i - 1][0]
        times_i = times[i-1] #results[i - 1][1]
        # Vector of length all_times
        extended_scores_i = [worst_score] * len(extended_times)
        # Indices of current times in all_times vector
        indexes_i = [i in times_i for i in extended_times]
        indexes_i = np.where(indexes_i)[0]
        # Fill up an extended vector with the
        for x, y in zip(indexes_i, scores_i):
            extended_scores_i[x] = y
        # extended_scores_i[indexes_i] = scores_i
        print(extended_scores_i)
        extended_scores_i = np.maximum.accumulate(extended_scores_i)
        # Update results
        extended_scores.append(extended_scores_i)

        print(extended_scores_i)

    # Take averages
    score_avg = np.mean(extended_scores, axis=0)
    score_std = np.std(extended_scores, axis=0) #/ (NB_ORACLES ** 0.5)

    print(score_avg, score_std)
    exit()




#avg_final_rew("4x0", "2a", 3, 1)
avg_final_rew(oracle="4x0", approach="2a_viper", nb_seeds=15, depth=2)
avg_final_rew(oracle="32x0", approach="2a_noMax", nb_seeds=15, depth=2)
avg_final_rew(oracle="32x0", approach="2a_max", nb_seeds=15, depth=2)

import numpy as np
import argparse
import matplotlib.pyplot as plt
from shutil import copyfile


def avg_final_rew(oracle, approach, nb_seeds, depth):
    rew_final = []
    for i in range(nb_seeds):
        load_from = './Oracle/' + str(oracle) + '/' + str(i+1) + '/' + approach + "/TimeVsReward_" + str(depth) + '.npy'
        rew_final.append(np.load(load_from)[-1][1])
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
        t_i = [item[0] for item in times_and_rewards]
        r_i = [item[1] for item in times_and_rewards]

        rewards.append(r_i)
        times.append(t_i)

        extended_times.extend(t_i)
        results.append(times_and_rewards)
    extended_times.sort()

    worst_score = -10e10

    # Extend score vectors to be consistent with extended times
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
        extended_scores_i = np.maximum.accumulate(extended_scores_i)
        # Update results
        extended_scores.append(extended_scores_i)

    # Take averages
    score_avg = np.mean(extended_scores, axis=0)
    score_std = np.std(extended_scores, axis=0) #/ (NB_ORACLES ** 0.5)

    return extended_times, score_avg, score_std


def avg_rew_vs_rollout(oracle, approach, nb_seeds, depth):

    rewards = []
    for i in range(nb_seeds):
        load_from = './Oracle/' + str(oracle) + '/' + str(i+1) + '/' + approach + "/TimeVsReward_" + str(depth) + '.npy'
        times_and_rewards = np.load(load_from)
        r_i = [item[1] for item in times_and_rewards]
        rewards.append(r_i)

    score_avg = np.mean(rewards, axis=0)
    score_std = np.std(rewards, axis=0) #/ (NB_ORACLES ** 0.5)

    return range(len(score_avg)), score_avg, score_std



config_a2 = [["4x0", "2a", 15, 2], ["32x0", "2a", 15, 2], ["256x0", "2a", 15, 2], ["64x64", "2a", 15, 2], ["256x256", "2a", 15, 2]]
config_a3 = [["4x0", "2a", 15, 3], ["32x0", "2a", 15, 3], ["256x0", "2a", 15, 3], ["64x64", "2a", 15, 3], ["256x256", "2a", 15, 3]]


configs = config_a2# + config_a3

# Generate Plots
for i, config in enumerate(configs):
    #extended_times, score_avg, score_std = avg_rew_vs_time(config[0], config[1], config[2], config[3])
    extended_times, score_avg, score_std = avg_rew_vs_rollout(config[0], config[1], config[2], config[3])
    plt.plot(extended_times, score_avg, label=config[0]+config[1]+str(config[3]))
    plt.fill_between(extended_times, score_avg - score_std, score_avg + score_std, alpha=0.2)

plt.xlabel('Runtime (s)')
plt.ylabel('Reward')
plt.title("CartPole-v1")
plt.ylim([-200, 250])
plt.legend(loc='lower right')
plt.savefig("plot_2_roll.png", dpi=1080, bbox_inches="tight")
plt.pause(10)


# AugTree (2)
print("Depth 2")
avg_final_rew(oracle="4x0", approach="2a", nb_seeds=15, depth=2)
avg_final_rew(oracle="32x0", approach="2a", nb_seeds=15, depth=2)
avg_final_rew(oracle="256x0", approach="2a", nb_seeds=15, depth=2)
avg_final_rew(oracle="64x64", approach="2a", nb_seeds=15, depth=2)
avg_final_rew(oracle="256x256", approach="2a", nb_seeds=15, depth=2)
print("-------------")

# AugTree (3)
print("Depth 3")
avg_final_rew(oracle="4x0", approach="2a", nb_seeds=15, depth=3)
avg_final_rew(oracle="32x0", approach="2a", nb_seeds=15, depth=3)
avg_final_rew(oracle="256x0", approach="2a", nb_seeds=15, depth=3)
avg_final_rew(oracle="64x64", approach="2a", nb_seeds=15, depth=3)
avg_final_rew(oracle="256x256", approach="2a", nb_seeds=15, depth=3)
print("-------------")


print("2")
avg_final_rew(oracle="4x0", approach="2b", nb_seeds=15, depth=2)
avg_final_rew(oracle="32x0", approach="2b", nb_seeds=15, depth=2)
avg_final_rew(oracle="256x0", approach="2b", nb_seeds=15, depth=2)
avg_final_rew(oracle="64x64", approach="2b", nb_seeds=15, depth=2)
avg_final_rew(oracle="256x256", approach="2b", nb_seeds=15, depth=2)

print("3")
avg_final_rew(oracle="4x0", approach="2b", nb_seeds=15, depth=3)
avg_final_rew(oracle="32x0", approach="2b", nb_seeds=15, depth=3)
avg_final_rew(oracle="256x0", approach="2b", nb_seeds=15, depth=3)
avg_final_rew(oracle="64x64", approach="2b", nb_seeds=15, depth=3)
avg_final_rew(oracle="256x256", approach="2b", nb_seeds=15, depth=3)

print("5")
avg_final_rew(oracle="4x0", approach="2b", nb_seeds=15, depth=5)
avg_final_rew(oracle="32x0", approach="2b", nb_seeds=15, depth=5)
avg_final_rew(oracle="256x0", approach="2b", nb_seeds=15, depth=5)
avg_final_rew(oracle="64x64", approach="2b", nb_seeds=15, depth=5)
avg_final_rew(oracle="256x256", approach="2b", nb_seeds=15, depth=5)

print("6")
avg_final_rew(oracle="4x0", approach="2b", nb_seeds=15, depth=6)
avg_final_rew(oracle="32x0", approach="2b", nb_seeds=15, depth=6)
avg_final_rew(oracle="256x0", approach="2b", nb_seeds=15, depth=6)
avg_final_rew(oracle="64x64", approach="2b", nb_seeds=15, depth=6)
avg_final_rew(oracle="256x256", approach="2b", nb_seeds=15, depth=6)

print("Depth 8")
avg_final_rew(oracle="4x0", approach="2b", nb_seeds=15, depth=8)
avg_final_rew(oracle="32x0", approach="2b", nb_seeds=15, depth=8)
avg_final_rew(oracle="256x0", approach="2b", nb_seeds=15, depth=8)
avg_final_rew(oracle="64x64", approach="2b", nb_seeds=15, depth=8)
avg_final_rew(oracle="256x256", approach="2b", nb_seeds=15, depth=8)



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
    std_Y = np.std(rew_final, axis=0)
    print(oracle, approach, depth, ":", mean_Y, std_Y)
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
    score_std = np.std(extended_scores, axis=0)

    return extended_times, score_avg, score_std


def avg_rew_vs_rollout(oracle, approach, nb_seeds, depth):

    rewards = []
    for i in range(nb_seeds):
        load_from = './Oracle/' + str(oracle) + '/' + str(i+1) + '/' + approach + "/TimeVsReward_" + str(depth) + '.npy'
        times_and_rewards = np.load(load_from)
        r_i = [item[1] for item in times_and_rewards]
        rewards.append(r_i)
    score_avg = np.mean(rewards, axis=0)

    return range(len(score_avg)), np.mean(rewards, axis=0), np.std(rewards, axis=0)



configs_a2 = [["32x0", "2a_VIPER_WEIGHT_FINAL", 15, 2], ["256x0", "2a_VIPER_WEIGHT_FINAL", 15, 2], ["64x64", "2a_VIPER_WEIGHT_FINAL", 15, 2], ["256x256", "2a_VIPER_WEIGHT_FINAL", 15, 2]]
configs_a3 = [["32x0", "2a_VIPER_WEIGHT_FINAL", 15, 3], ["256x0", "2a_VIPER_WEIGHT_FINAL", 15, 3], ["64x64", "2a_VIPER_WEIGHT_FINAL", 15, 3], ["256x256", "2a_VIPER_WEIGHT_FINAL", 15, 3]]



configs_b7 = [["256x0", "2b_VIPER_WEIGHT_FINAL", 15, 7]]
configs_b8 = [["256x0", "2b_VIPER_WEIGHT_FINAL", 15, 8]]

configs = configs_a2 + configs_b7 + configs_b8 + [["32x0", "2a_NO_WEIGHT_FINAL", 15, 2], ["256x0", "2a_NO_WEIGHT_FINAL", 15, 2]]


# Oracle Size: 32x0
configs_aX_32x0 = [["32x0", "2a_VIPER_WEIGHT_FINAL", 15, 2], ["32x0", "2a_NO_WEIGHT_FINAL", 15, 3]]
configs_bX_32x0 = [["32x0", "2b_VIPER_WEIGHT_FINAL", 15, 2], ["32x0", "2b_VIPER_WEIGHT_FINAL", 15, 3],
                   ["32x0", "2b_VIPER_WEIGHT_FINAL", 15, 4], ["32x0", "2b_VIPER_WEIGHT_FINAL", 15, 5],
                   ["32x0", "2b_VIPER_WEIGHT_FINAL", 15, 6], ["32x0", "2b_VIPER_WEIGHT_FINAL", 15, 7],
                   ["32x0", "2b_VIPER_WEIGHT_FINAL", 15, 8], ["32x0", "2b_VIPER_WEIGHT_FINAL", 15, 9]]
                   #["32x0", "2b_VIPER_WEIGHT_FINAL", 15, 10]]
configs_32x0 = [configs_aX_32x0, configs_bX_32x0]

# Oracle Size: 256x0
configs_aX_256x0 = [["256x0", "2a_VIPER_WEIGHT_FINAL", 15, 2], ["256x0", "2a_NO_WEIGHT_FINAL", 15, 3]]
configs_bX_256x0 = [["256x0", "2b_VIPER_WEIGHT_FINAL", 15, 2], ["256x0", "2b_VIPER_WEIGHT_FINAL", 15, 3],
                    ["256x0", "2b_VIPER_WEIGHT_FINAL", 15, 4], ["256x0", "2b_VIPER_WEIGHT_FINAL", 15, 5],
                    ["256x0", "2b_VIPER_WEIGHT_FINAL", 15, 6], ["256x0", "2b_VIPER_WEIGHT_FINAL", 15, 7],
                    ["256x0", "2b_VIPER_WEIGHT_FINAL", 15, 8], ["256x0", "2b_VIPER_WEIGHT_FINAL", 15, 9],
                    ["256x0", "2b_VIPER_WEIGHT_FINAL", 15, 10]]
configs_256x0 = [configs_aX_256x0, configs_bX_256x0]

# Oracle Size: 64x64
configs_aX_64x64 = [["64x64", "2a_VIPER_WEIGHT_FINAL", 15, 2], ["64x64", "2a_NO_WEIGHT_FINAL", 15, 3]]
configs_bX_64x64 = [["64x64", "2b_VIPER_WEIGHT_FINAL", 15, 2], ["64x64", "2b_VIPER_WEIGHT_FINAL", 15, 3],
                    ["64x64", "2b_VIPER_WEIGHT_FINAL", 15, 4], ["64x64", "2b_VIPER_WEIGHT_FINAL", 15, 5],
                    ["64x64", "2b_VIPER_WEIGHT_FINAL", 15, 6], ["64x64", "2b_VIPER_WEIGHT_FINAL", 15, 7],
                    ["64x64", "2b_VIPER_WEIGHT_FINAL", 15, 8], ["64x64", "2b_VIPER_WEIGHT_FINAL", 15, 9],
                    ["64x64", "2b_VIPER_WEIGHT_FINAL", 15, 10]]
configs_64x64 = [configs_aX_64x64, configs_bX_256x0] #, configs_bX_64x64]

# Oracle Size: 256x256
configs_aX_256x256 = [["256x256", "2a_VIPER_WEIGHT_FINAL", 15, 2], ["256x256", "2a_VIPER_WEIGHT_FINAL", 15, 3]]
configs_bX_256x256 = [["256x256", "2b_VIPER_WEIGHT_FINAL", 15, 2], ["256x256", "2b_VIPER_WEIGHT_FINAL", 15, 3],
                      ["256x256", "2b_VIPER_WEIGHT_FINAL", 15, 4], ["256x256", "2b_VIPER_WEIGHT_FINAL", 15, 5],
                      ["256x256", "2b_VIPER_WEIGHT_FINAL", 15, 6], ["256x256", "2b_VIPER_WEIGHT_FINAL", 15, 7],
                      ["256x256", "2b_VIPER_WEIGHT_FINAL", 15, 8], ["256x256", "2b_VIPER_WEIGHT_FINAL", 15, 9],
                      ["256x256", "2b_VIPER_WEIGHT_FINAL", 15, 10]]
configs_256x256 = [configs_aX_256x256, configs_bX_256x256]


configs = configs_bX_256x0 + configs_aX_256x0


# REWARD VS DEPTH
fig, axs = plt.subplots(1, 4)

for j, configs in enumerate([configs_32x0, configs_256x0, configs_64x64]):  #, configs_256x256]):
    label = "AugTree"
    for a, approach in enumerate(configs):
        avg_scores = []
        std_scores = []
        depths = []
        # Approach a
        for i, config in enumerate(approach):
            score_avg, score_std = avg_final_rew(config[0], config[1], config[2], config[3])
            depths.append(config[3])
            avg_scores.append(score_avg)
            std_scores.append(score_std)
        # Plot approach a
        axs[j].errorbar(depths, avg_scores, std_scores, label=label)
        label = "VIPER"
    # Final Plot
    axs[j].axhline(y=200., color='g', linestyle='--', label="Solved Threshold")
    axs[j].axhline(y=234., color='r', linestyle='--', label="Oracle") # FIX THIS
    axs[j].set_title("Architecture: [" + config[0] + "]")

    #axs[j].ylim([-100, 300])

plt.legend(loc='lower right')
#fig.suptitle('LunarLander-v2')

for ax in axs.flat:
    ax.set(xlabel='Tree Depth', ylabel='Reward')
    ax.label_outer()

plt.savefig("plot_RewVsDepth.png", dpi=1080, bbox_inches="tight")


plt.pause(60)
plt.clf()
exit()


# REWARD VS DEPTH
label = "AugTree"
for a, approach in enumerate(configs_256):
    avg_scores = []
    std_scores = []
    depths = []
    # Approach a
    for i, config in enumerate(approach):
        score_avg, score_std = avg_final_rew(config[0], config[1], config[2], config[3])
        depths.append(config[3])
        avg_scores.append(score_avg)
        std_scores.append(score_std)
    # Plot approach a
    plt.errorbar(depths, avg_scores, std_scores, label=label) #config[1][0:2])
    label = "VIPER"
# Final Plot
plt.axhline(y=200., color='g', linestyle='--', label="Solve Threshold")
plt.axhline(y=234., color='r', linestyle='--', label="Oracle")
plt.xlabel('Tree Depth')
plt.ylabel('Reward')
plt.title("LunarLander-v2")
plt.ylim([-100, 300])
plt.legend(loc='lower right')
plt.savefig("plot_RewVsDepth.png", dpi=1080, bbox_inches="tight")
plt.pause(10)
plt.clf()
exit()


# REWARD VS ROLLOUT
for i, config in enumerate(configs):
    #extended_times, score_avg, score_std = avg_rew_vs_time(config[0], config[1], config[2], config[3])
    extended_times, score_avg, score_std = avg_rew_vs_rollout(config[0], config[1], config[2], config[3])
    plt.plot(extended_times, score_avg, label=config[0]+config[1][0:3]+str(config[3]))
    plt.fill_between(extended_times, score_avg - score_std, score_avg + score_std, alpha=0.2)
    avg_final_rew(config[0], config[1], config[2], config[3])
plt.axhline(y=200., color='b', linestyle='--')
plt.xlabel('DAgger Rollouts')
plt.ylabel('Reward')
plt.title("LunarLander-v2")
plt.ylim([-300, 300])
plt.legend(loc='lower right')
plt.savefig("plot_RewVsRoll.png", dpi=1080, bbox_inches="tight")
plt.pause(1)
plt.clf()


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



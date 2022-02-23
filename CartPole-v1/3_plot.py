import numpy as np
import matplotlib.pyplot as plt


def avg_final_rew(oracle, approach, nb_seeds, depth):
    rew_final = []
    for i in range(nb_seeds):
        load_from = './Oracle/' + str(oracle) + '/' + str(i+1) + '/' + approach + "/TimeVsReward_" + str(depth) + '.npy'
        rew_final.append(np.load(load_from)[-1][1])
        print(rew_final[-1])
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
        #print(times)
        extended_times.extend(t_i)
        results.append(times_and_rewards)
    extended_times.sort()


    worst_score = 0 # min(rewards)

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
#        #print(extended_scores_i)
        extended_scores_i = np.maximum.accumulate(extended_scores_i)
        # Update results
        extended_scores.append(extended_scores_i)

        #print(extended_scores_i)

    # Take averages
    score_avg = np.mean(extended_scores, axis=0)
    score_std = np.std(extended_scores, axis=0) #/ (NB_ORACLES ** 0.5)

    return extended_times, score_avg, score_std


#avg_final_rew("32x0", "2a_")




configs = [["256x0", "2c", 15, 1], ["64x64", "2c", 15, 1], ["256x256", "2c", 15, 1]]
configs = [["1x0", "2a_viper", 15, 1], ["32x0", "2a", 15, 1], ["64x64", "2a", 15, 1], ["256x256", "2a", 15, 1]]


# Generate Tables
for i, config in enumerate(configs):
    avg_final_rew(config[0], config[1], config[2], config[3])


# Generate Plots
for i, config in enumerate(configs):
    extended_times, score_avg, score_std = avg_rew_vs_time(config[0], config[1], config[2], config[3])
    plt.plot(extended_times, score_avg, label=config[0])
    plt.fill_between(extended_times, score_avg - score_std, score_avg + score_std, alpha=0.2)

plt.xlabel('Runtime (s)')
plt.ylabel('Reward')
plt.title("CartPole-v1")
plt.ylim([0, 510])
plt.savefig("plot_2a.png", dpi=1080, bbox_inches="tight")
plt.legend(loc='lower right')
plt.pause(10)


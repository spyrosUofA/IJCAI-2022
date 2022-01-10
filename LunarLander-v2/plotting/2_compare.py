import numpy as np
import matplotlib.pyplot as plt

configs = ['D000', 'D010', 'D100', 'D110', 'E010_D010']
names = ["NDPS", "AugNDPS", "NDPS_BO", "AugNDPS_BO", "AugProg", "AugSketch"]

#configs = ['D010b', 'D010', 'D000']

for count, config in enumerate(configs):

    time = np.load('./plot_data/' + config + '_time.npy')
    r_avg = np.load('./plot_data/' + config + '_avg.npy')
    r_std = np.load('./plot_data/' + config + '_std.npy')

    plt.plot(time, r_avg)
    plt.fill_between(time, r_avg - r_std, r_avg + r_std, alpha=0.2)

plt.xlabel('Sequential Runtime (s)')
plt.ylabel('Reward')
plt.title('LunarLander-v2 (without BayesOpt)')
plt.ylim([-200, 250])
#plt.legend(['Aug 32', 'Aug 64', 'NDPS'], loc='lower right')
plt.legend(names, loc='lower right')
plt.show()

#plt.savefig('./plots/COMPARE.png', dpi=1080, bbox_inches="tight")


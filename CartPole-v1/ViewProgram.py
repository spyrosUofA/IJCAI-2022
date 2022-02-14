import numpy as np
from sklearn import tree
import pandas as pd
import re
import pickle


def relu_string(relus):
    relu_names = []
    for i, relu in enumerate(relus):
        name = '(' + str(np.around(relu[0], 2)) + " *dot* obs[:] + " + str(np.round(relu[1], 2)) + ")"
        relu_names.append(name)
    return relu_names




rews = np.load("./Oracle/4x0/1/2a/Rew_1.npy").tolist()
print(rews)

relus = pickle.load(open("./Oracle/4x0/1/ReLUs.pkl", "rb"))
relu_names = relu_string(relus)
#relu_names = ["w" + str(i).zfill(1) for i in range(4)]
relu_names.extend(["x", "v_x", "theta", "v_th"])

trees = pickle.load(open("./Oracle/4x0/1/2a/Program_1.pkl", "rb"))

regr_1 = trees
tree_rules = tree.export_text(regr_1, feature_names=relu_names)

print(tree_rules)


# Extract decision rules as strings
decision_rules = tree_rules.replace("|--- class:", "act = ")
decision_rules = decision_rules.replace("|---", "if")
decision_rules = decision_rules.replace("|", "")

print(decision_rules)
boolean_rules = re.findall(r'if (.*)', decision_rules)



def avg_rew_vs_rolloout(oracle, approach, nb_seeds, depth):

    rewards = []
    for i in range(nb_seeds):
        load_from = './Oracle/' + str(oracle) + '/' + str(i+1) + '/' + approach + "/TimeVsReward_" + str(depth) + '.npy'
        times_and_rewards = np.load(load_from)
        r_i = [item[1] for item in times_and_rewards]
        rewards.append(r_i)

    score_avg = np.mean(rewards, axis=0)
    score_std = np.std(rewards, axis=0) #/ (NB_ORACLES ** 0.5)

    return range(len(score_avg)), score_avg, score_std
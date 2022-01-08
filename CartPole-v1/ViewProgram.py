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




rews = np.load("Oracle/4x0/2/AugTreeRewards.npy").tolist()
print(rews)

relus = pickle.load(open("Oracle/4x0/2/ReLUs.pkl", "rb"))
#relu_names = relu_string(relus)
relu_names = ["w" + str(i).zfill(1) for i in range(4)]
relu_names.extend(["x", "y", "v_x", "v_y", "theta", "v_th", "c_l", "c_r"])

trees = pickle.load(open("Oracle/4x0/2/AugTreePrograms.pkl", "rb"))[0]

regr_1 = trees
tree_rules = tree.export_text(regr_1) #, feature_names=relu_names)

print(tree_rules)

print(relus[0])
print(relus[2])
print(relus[3])




# Extract decision rules as strings
decision_rules = tree_rules.replace("|--- class:", "act = ")
decision_rules = decision_rules.replace("|---", "if")
decision_rules = decision_rules.replace("|", "")

print(decision_rules)
boolean_rules = re.findall(r'if (.*)', decision_rules)


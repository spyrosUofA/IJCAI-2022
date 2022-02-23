import numpy as np
import pickle
import gym
from sklearn import tree


def relu_string(relus):
    names = []
    for _, relu in enumerate(relus):
        name = '(' + str(np.around(relu[0], 2)) + " *dot* obs[:] + " + str(np.round(relu[1], 2)) + ")"
        names.append(name)
    return names


def prepare_program(dec_tree, relus):
    used_features = list(set([x for x in dec_tree.tree_.feature if 0 <= x < len(relus)]))
    used_relus = []
    for j in used_features:
        used_relus.append(relus[j])
    return used_features, used_relus


def my_program(obs, dec_tree, nb_relus, used_features, used_relus):
    all_features = [0] * nb_relus
    all_features.extend(obs)
    for i, relu in enumerate(used_relus):
        all_features[used_features[i]] = max(0, np.dot(relu[0], obs) + relu[1])
    return dec_tree.predict([all_features])[0]


# SPECIFY POLICY
oracle = "1x0"
method = "2a_viper"
depth = "1"
seed = "3"





# LOAD POLICY
policy = pickle.load(open("./Oracle/" + oracle + '/' + seed + '/' + method + '/Program_' + depth + '.pkl', "rb"))
relus = pickle.load(open("./Oracle/" + oracle + '/' + seed + "/ReLUs.pkl", "rb"))
relu_names = relu_string(relus)
relu_names.extend(["x_c", "v_c", "c_L", "c_R"])
nb_relus = len(relus)
used_features, used_relus = prepare_program(policy, relus)
print(tree.export_text(policy, feature_names=relu_names))


# SETUP ENVIRONMENT
env = gym.make("CartPole-v1")
env.seed(0)
ob = env.reset()
averaged = 0.0
games = 100


n0 = [-0.09,  0.12, -0.12,  0.27,  0.22,  0.18,  0.22,  0.1, 0]
n10 = [ 0.04,  0.08,  0.2,  0.2,  -0.24, -0.26,  0.21,  0.15, 0.01]
n11 = [0, 0, 0, 0, 0, 0, -1, -1, -1]


# EVALUATE POLICY
for g in range(games):
    ob = env.reset()
    reward = 0.0
    done = False
    while not done:
        #env.render()
        action = my_program(ob, policy, nb_relus, used_features, used_relus)
        #print(ob[-2:], action)
        ob, r_t, done, _ = env.step(action)
        reward += r_t
    print(reward)
    averaged += reward
    env.close()
averaged /= games

print(games, averaged)

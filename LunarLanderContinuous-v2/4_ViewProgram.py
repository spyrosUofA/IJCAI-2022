import numpy as np
import pickle
import gym
from sklearn import tree


def get_neurons(obs, relus):
    neurons = []
    for i, relu in enumerate(relus):
        neuron = max(0, np.dot(obs, relu[0]) + relu[1])
        neurons.append(neuron)
    neurons.extend(obs)
    return neurons


def my_program_slow(obs, tree, relus):
    neurons = get_neurons(obs, relus)
    return tree.predict([neurons])[0]


def prepare_program(dec_tree, relus):
    used_features = list(set([x for x in dec_tree.tree_.feature if 0 <= x < len(relus)]))
    used_relus = []
    for i in used_features:
        used_relus.append(relus[i])
    return used_features, used_relus


def my_program(obs, dec_tree, nb_relus, used_features, used_relus):
    all_features = [0] * nb_relus
    all_features.extend(obs)
    for i, relu in enumerate(used_relus):
        all_features[used_features[i]] = max(0, np.dot(relu[0], obs) + relu[1])
    return dec_tree.predict([all_features])[0]


policy = pickle.load(open("./Oracle/256x0/15/2a/Program_2.pkl", "rb"))
relus = pickle.load(open("./Oracle/256x0/15/ReLUs.pkl", "rb"))

# SETUP ENVIRONMENT
nb_relus = len(relus)
used_features, used_relus = prepare_program(policy, relus)
print(tree.export_text(policy))

env = gym.make("LunarLanderContinuous-v2")
env.seed(0)
ob = env.reset()
averaged = 0.0
games = 100

actions = list()

for i in range(games):
    ob = env.reset()
    reward = 0.0
    done = False
    while not done:
        #env.render()
        action = my_program(ob, policy, nb_relus, used_features, used_relus)
        actions.append(action[0])
        ob, r_t, done, _ = env.step(action)
        reward += r_t
    print(reward)
    actions = list(set(actions))
    averaged += reward
    env.close()
averaged /= games

print(games, averaged)
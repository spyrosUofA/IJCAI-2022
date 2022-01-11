import gym
from stable_baselines3 import PPO
from stable_baselines3.common.policies import obs_as_tensor
import torch
import os
import pickle
import numpy as np
import copy
from sklearn import tree
import multiprocessing
from itertools import repeat
from operator import itemgetter
from numpy.random import choice


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def forward_pass(obs, w1, b1, w2, b2, w3=None, b3=None):
    pass


def get_weights(model):
    pass


def get_weights_1(model):
    w1 = model.policy.state_dict()['mlp_extractor.policy_net.0.weight'].detach().numpy()
    b1 = model.policy.state_dict()['mlp_extractor.policy_net.0.bias'].detach().numpy()
    w2 = model.policy.state_dict()['action_net.weight'].detach().numpy()
    b2 = model.policy.state_dict()['action_net.bias'].detach().numpy()
    return w1, b1, w2, b2


def forward_pass_1(obs, w1, b1, w2, b2):
    # Propagate forward
    l1_neurons = np.maximum(0, np.matmul(w1, obs) + b1)
    outputs = np.matmul(l1_neurons, np.transpose(w2)) + b2
    # Action probabilities
    probs = softmax(outputs)
    best_action = np.argmax(probs)
    viper_weight = max(probs) - min(probs)
    return best_action, viper_weight, np.append(l1_neurons, obs)


def forward_pass_2(obs, w1, b1, w2, b2, w3, b3):
    pass


def initialize_history(env, model, save_to, games, get_weights):
    observations = []
    actions = []
    viper_weights = []
    neurons = []
    r = 0.0

    w1, b1, w2, b2 = get_weights(model)

    for episode in range(games):
        state = env.reset()
        done = False
        while not done:
            # Query oracle
            action, viper_weight, l1_neurons = forward_pass_1(state, w1, b1, w2, b2)
            # Record Trajectory
            observations.append(state)
            actions.append(action)
            viper_weights.append(viper_weight)
            neurons.append(l1_neurons)
            # Interact with Environment
            state, reward, done, _ = env.step(action)
            r += reward
    print(r)
    exit()
    # Save Data
    np.save(file=save_to + 'Observations_' + str(games) + '.npy', arr=observations)
    np.save(file=save_to + 'Actions_' + str(games) + '.npy', arr=actions)
    np.save(file=save_to + 'ViperWeights_' + str(games) + '.npy', arr=viper_weights)
    np.save(file=save_to + 'Neurons_' + str(games) + '.npy', arr=neurons)
    np.savetxt(save_to + 'OracleReward.txt', r)


def augmented_dagger(env, model, save_to, depth, rollouts, eps_per_rollout, seed):

    X = np.load(save_to + "Neurons_" + str(eps_per_rollout) + ".npy").tolist()
    Y = np.load(save_to + "Actions_" + str(eps_per_rollout) + ".npy").tolist()
    VW = np.load(save_to + 'ViperWeights_' + str(eps_per_rollout) + ".npy").tolist()
    relu_programs = pickle.load(open(save_to + "ReLUs.pkl", "rb"))
    best_reward = -10000
    regr_tree = tree.DecisionTreeClassifier(max_depth=depth, random_state=seed)
    w1, b1, w2, b2 = get_weights_1(model)

    for r in range(rollouts):

        # Resample dataset (VIPER)
        #draw = choice(range(len(Y)), 100000, p=softmax(VW))
        #x = [X[i] for i in draw]
        #y = [Y[i] for i in draw]
        #print(len(Y))

        # Fit decision tree
        regr_tree.fit(X, Y)

        # Rollout policy
        steps = 0
        reward_avg = 0
        for i in range(eps_per_rollout):
            ob = env.reset()
            done = False
            while not done:
                # Query oracle
                a_star, viper_weight, l1_neurons = forward_pass_1(ob, w1, b1, w2, b2)
                # DAgger
                X.append(l1_neurons)
                Y.append(a_star)
                VW.append(viper_weight)
                # Interact with Environment
                action = regr_tree.predict([l1_neurons])[0]
                ob, r_t, done, _ = env.step(action)
                steps += 1
                reward_avg += r_t
        print(r, reward_avg / eps_per_rollout)

        # Evaluate over 100 consecutive episodes
        for i in range(25 - eps_per_rollout):
            ob = env.reset()
            done = False
            while not done:
                l1_neurons = np.maximum(0, np.matmul(w1, ob) + b1)
                l1_neurons.extend(ob)
                action = regr_tree.predict([l1_neurons])[0]
                ob, r_t, done, _ = env.step(action)
                reward_avg += r_t
        #reward_avg = reward_avg / 100.
        #print(r, reward_avg)

        # Update best program
        if reward_avg > best_reward:
            best_reward = reward_avg
            best_program = copy.deepcopy(regr_tree)




    return best_reward, best_program


def main(seed, l1_actor, l2_actor):

    # configure directory
    save_to = './Oracle/' + str(l1_actor) + 'x' + str(l2_actor) + '/' + str(seed) + '/'
    print(save_to)
    if not os.path.exists(save_to):
        os.makedirs(save_to)

    if l2_actor == 0:
        net_arch = [l1_actor]



    else:
        net_arch = [l1_actor, l2_actor]

    # create environment
    env = gym.make("LunarLander-v2")
    env.seed(seed)

    # load oracle
    model = PPO('MlpPolicy', env, seed=seed, verbose=0, policy_kwargs=dict(net_arch=[dict(pi=net_arch, vf=[128, 128])], activation_fn=torch.nn.ReLU))
    model = model.load(save_to + 'model')

    # generate experience
    initialize_history(env, model, save_to, 25)

    # augmented DAgger rollouts
    augmented_dagger(env, model, save_to, 2, 25, 25, seed)

if __name__ == "__main__":

    main(17, 4, 0)
    exit()

    pool = multiprocessing.Pool(15)
    pool.starmap(main, zip(range(1, 16), repeat(32), repeat(0)))
    #pool.starmap(main, zip(range(16, 30), repeat(32), repeat(0)))
    #pool.starmap(main, zip(range(16, 30), repeat(256), repeat(0)))
    #pool.starmap(main, zip(range(16, 30), repeat(64), repeat(64)))
    #pool.starmap(main, zip(range(16, 30), repeat(256), repeat(256)))
import gym
from stable_baselines3 import PPO
import torch
import os
import pickle
import numpy as np
import copy
from sklearn import tree
import multiprocessing
from itertools import repeat
import time


def initialize_history(env, model, load_from, games, get_weights, forward_pass):
    observations = []
    actions = []
    viper_weights = []
    neurons = []
    r = 0.0
    w1, b1, w2, b2, w3, b3 = get_weights(model)

    for episode in range(games):
        state = env.reset()
        done = False
        while not done:
            # Query oracle
            action, viper_weight, l1_neurons = forward_pass(state, w1, b1, w2, b2, w3, b3)
            # Record Trajectory
            observations.append(state)
            actions.append(action)
            viper_weights.append(viper_weight)
            l1_neurons.extend(state)
            neurons.append(l1_neurons)
            # Interact with Environment
            state, reward, done, _ = env.step(action)
            r += reward
    r = r / games
    print("Oracle Reward:", r)
    np.savetxt(load_from + 'OracleReward.txt', [r])

    return neurons, actions, viper_weights



def main(seed, l1_actor, l2_actor, depth):

    # configure directory
    load_from = './Oracle/' + str(l1_actor) + 'x' + str(l2_actor) + '/' + str(seed) + '/'
    save_to = load_from + '2a/'
    if not os.path.exists(save_to):
        os.makedirs(save_to)

    # configure neural policy
    if l2_actor == 0:
        net_arch = [l1_actor]

    else:
        net_arch = [l1_actor, l2_actor]

    # create environment
    env = gym.make("CartPole-v1")
    env.seed(seed)

    # load oracle
    #model = PPO('MlpPolicy', env, seed=seed, verbose=0, policy_kwargs=dict(net_arch=[dict(pi=net_arch, vf=[128, 128])], activation_fn=torch.nn.ReLU))
    model = PPO.load(load_from + 'model')

    r = 0.0

    for episode in range(100):
        state = env.reset()
        done = False
        while not done:
            # Query oracle
            action, mode
            # Record Trajectory
            observations.append(state)
            actions.append(action)
            viper_weights.append(viper_weight)
            l1_neurons.extend(state)
            neurons.append(l1_neurons)
            # Interact with Environment
            state, reward, done, _ = env.step(action)
            r += reward
    r = r / games
    print("Oracle Reward:", r)
    np.savetxt(load_from + 'OracleReward.txt', [r])



if __name__ == "__main__":

    main(4, 256, 0, 2)
    exit()
    # Depth 1
    for seed in range(1, 16):
        main(seed, 4, 0, 1)
        main(seed, 32, 0, 1)
        main(seed, 64, 64, 1)
        main(seed, 256, 256, 1)

    # Depth 2
    for seed in range(1, 16):
        main(seed, 4, 0, 2)
        main(seed, 32, 0, 2)
        main(seed, 64, 64, 2)
        main(seed, 256, 256, 2)

    #pool = multiprocessing.Pool(10)
    #pool.starmap(main, zip(range(1, 31), repeat(4), repeat(0)))
    #pool.starmap(main, zip(range(16, 31), repeat(32), repeat(0)))
    #pool.starmap(main, zip(range(16, 31), repeat(256), repeat(0)))
    #pool.starmap(main, zip(range(16, 31), repeat(64), repeat(64)))
    #pool.starmap(main, zip(range(16, 31), repeat(256), repeat(256)))
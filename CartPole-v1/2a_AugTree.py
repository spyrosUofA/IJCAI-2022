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


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


# Defining function for NN with 1 hidden layer
def get_weights_1(model):
    w1 = model.policy.state_dict()['mlp_extractor.policy_net.0.weight'].detach().numpy()
    b1 = model.policy.state_dict()['mlp_extractor.policy_net.0.bias'].detach().numpy()
    w2 = model.policy.state_dict()['action_net.weight'].detach().numpy()
    b2 = model.policy.state_dict()['action_net.bias'].detach().numpy()
    return w1, b1, w2, b2, None, None


def forward_pass_1(obs, w1, b1, w2, b2, w3=None, b3=None):
    # Propagate forward
    l1_neurons = np.maximum(0, np.matmul(w1, obs) + b1).tolist()
    outputs = np.matmul(l1_neurons, np.transpose(w2)) + b2
    # Action probabilities
    probs = softmax(outputs)
    log_probs = np.log(probs)
    best_action = np.argmax(probs)
    viper_weight = max(log_probs) - min(log_probs)
    return best_action, viper_weight, l1_neurons


def get_weights_2(model):
    w1 = model.policy.state_dict()['mlp_extractor.policy_net.0.weight'].detach().numpy()
    b1 = model.policy.state_dict()['mlp_extractor.policy_net.0.bias'].detach().numpy()
    w2 = model.policy.state_dict()['mlp_extractor.policy_net.2.weight'].detach().numpy()
    b2 = model.policy.state_dict()['mlp_extractor.policy_net.2.bias'].detach().numpy()
    w3 = model.policy.state_dict()['action_net.weight'].detach().numpy()
    b3 = model.policy.state_dict()['action_net.bias'].detach().numpy()
    return w1, b1, w2, b2, w3, b3


# Define function for NN with 2 hidden layers
def forward_pass_2(obs, w1, b1, w2, b2, w3, b3):
    # Propagate forward
    l1_neurons = np.maximum(0, np.matmul(w1, obs) + b1).tolist()
    l2_neurons = np.maximum(0, np.matmul(l1_neurons, np.transpose(w2)) + b2)
    outputs = np.matmul(l2_neurons, np.transpose(w3)) + b3
    # Action probabilities
    probs = softmax(outputs)
    log_probs = np.log(probs)
    best_action = np.argmax(probs)
    viper_weight = max(log_probs) - min(log_probs)
    return best_action, viper_weight, l1_neurons


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


def augmented_dagger(env, model, load_from, depth, rollouts, eps_per_rollout, seed, get_weights, forward_pass):

    t0 = time.time()

    # Instantiate loggers
    best_program = None
    best_reward = -10e10
    time_vs_reward = []

    # Setup task
    regr_tree = tree.DecisionTreeClassifier(max_depth=depth, random_state=seed)
    w1, b1, w2, b2, w3, b3 = get_weights(model)
    X, Y, VW = initialize_history(env, model, load_from, eps_per_rollout, get_weights, forward_pass)

    # Rollout N times
    for r in range(rollouts):

        # Resample dataset (VIPER)
        #draw = choice(range(len(Y)), 100000, p=softmax(VW))
        #x = [X[i] for i in draw]
        #y = [Y[i] for i in draw]
        #regr_tree.fit(x, y)

        # Fit decision tree
        regr_tree.fit(X, Y)

        # Evaluate over 100 consecutive episodes
        reward_avg = 0.0
        for i in range(100):
            ob = env.reset()
            done = False
            while not done:
                l1_neurons = np.maximum(0, np.matmul(w1, ob) + b1).tolist()
                l1_neurons.extend(ob)
                action = regr_tree.predict([l1_neurons])[0]
                ob, r_t, done, _ = env.step(action)
                reward_avg += r_t
        reward_avg /= 100.

        # Update best program
        if reward_avg > best_reward:
            best_reward = reward_avg
            best_program = copy.deepcopy(regr_tree)
            time_vs_reward.append([best_reward, time.time()-t0])
        print(r, best_reward)

        # Collect M trajectories
        for i in range(eps_per_rollout):
            ob = env.reset()
            done = False
            while not done:
                # Query oracle
                a_star, viper_weight, l1_neurons = forward_pass(ob, w1, b1, w2, b2, w3, b3)
                l1_neurons.extend(ob)
                # DAgger
                X.append(l1_neurons)
                Y.append(a_star)
                VW.append(viper_weight)
                # Interact with Environment
                action = best_program.predict([l1_neurons])[0]
                ob, r_t, done, _ = env.step(action)

    return best_reward, best_program, time_vs_reward


def main(seed, l1_actor, l2_actor, depth):

    # configure directory
    load_from = './Oracle/' + str(l1_actor) + 'x' + str(l2_actor) + '/' + str(seed) + '/'
    save_to = load_from + '2a/'
    if not os.path.exists(save_to):
        os.makedirs(save_to)

    # configure neural policy
    if l2_actor == 0:
        net_arch = [l1_actor]
        get_weights = get_weights_1
        forward_pass = forward_pass_1
    else:
        net_arch = [l1_actor, l2_actor]
        get_weights = get_weights_2
        forward_pass = forward_pass_2

    # create environment
    env = gym.make("CartPole-v1")
    env.seed(seed)

    # load oracle
    model = PPO('MlpPolicy', env, seed=seed, verbose=0, policy_kwargs=dict(net_arch=[dict(pi=net_arch, vf=[128, 128])], activation_fn=torch.nn.ReLU))
    model = model.load(load_from + 'model')

    # DAgger rollouts
    reward, program, time_vs_reward = augmented_dagger(env, model, load_from, depth, 25, 25, seed, get_weights, forward_pass)
    print(save_to)
    print("Depth: ", depth)
    print("Reward: ", reward)
    print(tree.export_text(program))

    # Save results
    np.save(file=save_to + 'Rew_' + str(depth) + '.npy', arr=reward)
    pickle.dump(program, file=open(save_to + 'Program_' + str(depth) + '.pkl', "wb"))
    np.save(file=save_to + 'TimeVsReward_' + str(depth) + '.npy', arr=time_vs_reward)


if __name__ == "__main__":

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
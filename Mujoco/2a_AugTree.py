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
import mujoco_py


# Defining function for NN with 1 hidden layer
def get_weights(model):
    w1 = model.policy.state_dict()['mlp_extractor.policy_net.0.weight'].detach().numpy()
    b1 = model.policy.state_dict()['mlp_extractor.policy_net.0.bias'].detach().numpy()
    return w1, b1


def forward_pass(obs, w1, b1, model):
    # Propagate forward
    l1_neurons = np.maximum(0, np.matmul(w1, obs) + b1).tolist()
    l1_neurons.extend(obs)
    a_star = model.policy.predict(obs, deterministic=True)[0]
    return a_star, l1_neurons


def initialize_history(env, model, games):
    actions = []
    neurons = []
    r = 0.0
    w1, b1 = get_weights(model)

    for episode in range(games):
        state = env.reset()
        done = False
        while not done:
            # Query oracle
            action, l1_neurons = forward_pass(state, w1, b1, model)
            # Record Trajectory
            neurons.append(l1_neurons)
            actions.append(action)
            # Interact with Environment
            state, reward, done, _ = env.step(action)
            r += reward
    r = r / games
    print("Oracle Reward:", r)

    return neurons, actions


def augmented_dagger(env, model, depth, rollouts, eps_per_rollout, seed, t0):

    # Instantiate loggers
    best_program = None
    best_reward = -10e10
    time_vs_reward = []

    # Setup task
    regr_tree = tree.DecisionTreeRegressor(max_depth=depth, random_state=seed)
    w1, b1 = get_weights(model)
    X, Y = initialize_history(env, model, eps_per_rollout)

    # Rollout N times
    for r in range(rollouts):

        # Fit decision tree
        regr_tree.fit(X, Y)

        # Collect M trajectories, aggregate dataset
        for i in range(eps_per_rollout):
            ob = env.reset()
            done = False
            while not done:
                # Query oracle
                a_star, l1_neurons = forward_pass(ob, w1, b1, model)
                # DAgger
                X.append(l1_neurons)
                Y.append(a_star)
                # Interact with Environment
                action = regr_tree.predict([l1_neurons])[0]
                ob, r_t, done, _ = env.step(action)

        # Evaluate over 100 consecutive episodes
        reward_avg = 0.0
        for i in range(100):
            ob = env.reset()
            done = False
            while not done:
                _, l1_neurons = forward_pass(ob, w1, b1, model)
                action = regr_tree.predict([l1_neurons])[0]
                ob, r_t, done, _ = env.step(action)
                reward_avg += r_t
        reward_avg /= 100.

        # Update best program
        if reward_avg > best_reward:
            best_reward = reward_avg
            best_program = copy.deepcopy(regr_tree)
        time_vs_reward.append([time.time()-t0, best_reward])
        print(r, best_reward)

    return best_reward, best_program, time_vs_reward


def main(seed, l1_actor, l2_actor, depth):

    t0 = time.time()

    # configure directory
    load_from = './Oracle/' + str(l1_actor) + 'x' + str(l2_actor) + '/' + str(seed) + '/'
    save_to = load_from + '2a/'
    if not os.path.exists(save_to):
        os.makedirs(save_to)

    # create environment
    env = gym.make('HalfCheetah-v3')
    env.seed(seed)

    # load oracle
    #model = PPO.load(load_from + 'model')
    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }
    model = PPO.load('./Oracle/256x256/0/HalfCheetah-v3', env=env, custom_objects=custom_objects)


    # DAgger rollouts
    reward, program, time_vs_reward = augmented_dagger(env, model, depth, 50, 25, seed, t0)
    print(save_to)
    print("Depth: ", depth)
    print("Reward: ", reward)
    print(tree.export_text(program))

    # Save results
    pickle.dump(program, file=open(save_to + 'Program_' + str(depth) + '.pkl', "wb"))
    np.save(file=save_to + 'TimeVsReward_' + str(depth) + '.npy', arr=time_vs_reward)
    print("Saved")


if __name__ == "__main__":

    main(0, 256, 256, 6)
    exit()


    # Multiprocess
    pool = multiprocessing.Pool(15)
    # Depth 3
    pool.starmap(main, zip(range(1, 16), repeat(4), repeat(0), repeat(3)))
    pool.starmap(main, zip(range(1, 16), repeat(32), repeat(0), repeat(3)))
    pool.starmap(main, zip(range(1, 16), repeat(256), repeat(0), repeat(3)))
    pool.starmap(main, zip(range(1, 16), repeat(64), repeat(64), repeat(3)))
    pool.starmap(main, zip(range(1, 16), repeat(256), repeat(256), repeat(3)))
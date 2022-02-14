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


def initialize_history(env, model, games):
    actions = []
    observations = []
    r = 0.0

    for episode in range(games):
        state = env.reset()
        done = False
        while not done:
            # Query oracle
            action = model.policy.predict(state, deterministic=True)[0]
            # Record Trajectory
            observations.append(state)
            actions.append(action)
            # Interact with Environment
            state, reward, done, _ = env.step(action)
            r += reward
    r = r / games
    print("Oracle Reward:", r)
    return observations, actions


def base_dagger(env, model, depth, rollouts, eps_per_rollout, seed, t0):

    # Instantiate loggers
    best_program = None
    best_reward = -10e10
    time_vs_reward = []

    # Setup task
    regr_tree = tree.DecisionTreeRegressor(max_depth=depth, random_state=seed)
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
                a_star = model.policy.predict(ob, deterministic=True)[0]
                # DAgger
                X.append(ob)
                Y.append(a_star)
                # Interact with Environment
                action = regr_tree.predict([ob])[0]
                ob, r_t, done, _ = env.step(action)

        # Evaluate over 100 consecutive episodes
        reward_avg = 0.0
        for i in range(100):
            ob = env.reset()
            done = False
            while not done:
                action = regr_tree.predict([ob])[0]
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
    save_to = load_from + '2b/'
    if not os.path.exists(save_to):
        os.makedirs(save_to)

    # create environment
    env = gym.make("LunarLanderContinuous-v2")
    env.seed(seed)

    # load oracle
    model = PPO.load(load_from + 'model')

    # DAgger rollouts
    reward, program, time_vs_reward = base_dagger(env, model, depth, 25, 25, seed, t0)
    print(save_to)
    print("Depth: ", depth)
    print("Reward: ", reward)
    print(tree.export_text(program))

    # Save results
    pickle.dump(program, file=open(save_to + 'Program_' + str(depth) + '.pkl', "wb"))
    np.save(file=save_to + 'TimeVsReward_' + str(depth) + '.npy', arr=time_vs_reward)
    print("Saved")


if __name__ == "__main__":

    # Multiprocess
    pool = multiprocessing.Pool(15)

    # Depth 2
    pool.starmap(main, zip(range(1, 16), repeat(4), repeat(0), repeat(2)))
    pool.starmap(main, zip(range(1, 16), repeat(32), repeat(0), repeat(2)))
    pool.starmap(main, zip(range(1, 16), repeat(256), repeat(0), repeat(2)))
    pool.starmap(main, zip(range(1, 16), repeat(64), repeat(64), repeat(2)))
    pool.starmap(main, zip(range(1, 16), repeat(256), repeat(256), repeat(2)))

    # Depth 3
    pool.starmap(main, zip(range(1, 16), repeat(4), repeat(0), repeat(3)))
    pool.starmap(main, zip(range(1, 16), repeat(32), repeat(0), repeat(3)))
    pool.starmap(main, zip(range(1, 16), repeat(256), repeat(0), repeat(3)))
    pool.starmap(main, zip(range(1, 16), repeat(64), repeat(64), repeat(3)))
    pool.starmap(main, zip(range(1, 16), repeat(256), repeat(256), repeat(3)))

    # Depth 5
    pool.starmap(main, zip(range(1, 16), repeat(4), repeat(0), repeat(5)))
    pool.starmap(main, zip(range(1, 16), repeat(32), repeat(0), repeat(5)))
    pool.starmap(main, zip(range(1, 16), repeat(256), repeat(0), repeat(5)))
    pool.starmap(main, zip(range(1, 16), repeat(64), repeat(64), repeat(5)))
    pool.starmap(main, zip(range(1, 16), repeat(256), repeat(256), repeat(5)))

    # Depth 6
    pool.starmap(main, zip(range(1, 16), repeat(4), repeat(0), repeat(6)))
    pool.starmap(main, zip(range(1, 16), repeat(32), repeat(0), repeat(6)))
    pool.starmap(main, zip(range(1, 16), repeat(256), repeat(0), repeat(6)))
    pool.starmap(main, zip(range(1, 16), repeat(64), repeat(64), repeat(6)))
    pool.starmap(main, zip(range(1, 16), repeat(256), repeat(256), repeat(6)))

    # Depth 8
    pool.starmap(main, zip(range(1, 16), repeat(4), repeat(0), repeat(8)))
    pool.starmap(main, zip(range(1, 16), repeat(32), repeat(0), repeat(8)))
    pool.starmap(main, zip(range(1, 16), repeat(256), repeat(0), repeat(8)))
    pool.starmap(main, zip(range(1, 16), repeat(64), repeat(64), repeat(8)))
    pool.starmap(main, zip(range(1, 16), repeat(256), repeat(256), repeat(8)))

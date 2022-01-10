import gym
from stable_baselines3 import PPO
import torch
import os
import pickle
import numpy as np


def main(l1_actor, l2_actor):
    r_avg = []

    for seed in range(1, 16):

        # load oracle
        load_from = './Oracle/' + str(l1_actor) + 'x' + str(l2_actor) + '/' + str(seed) + '/'
        model = PPO.load(load_from + 'model')

        # create environment
        env = gym.make("CartPole-v1")
        env.seed(seed)

        # run 100 episodes
        r = 0.0
        for episode in range(100):
            state = env.reset()
            done = False
            while not done:
                # Query oracle
                action = model.predict(state, deterministic=True)[0]
                # Interact with Environment
                state, reward, done, _ = env.step(action)
                r += reward
        r_avg.append(r / 100)

    mean_Y = np.mean(r_avg, axis=0)
    std_Y = np.std(r_avg, axis=0)  # * (nb_seeds ** -0.5)
    print(mean_Y, std_Y)

    np.savetxt('./Oracle/' + str(l1_actor) + 'x' + str(l2_actor) + '/Oracle_Rew.txt', [mean_Y, std_Y])


if __name__ == "__main__":
    #main(4, 0)
    main(32, 0)
    main(64, 64)
    main(256, 256)
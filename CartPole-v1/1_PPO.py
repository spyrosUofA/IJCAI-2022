import gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import torch
import os
import pickle
import multiprocessing
from itertools import repeat


def save_relus(model, save_to):
    relu_programs = []
    biases = model.policy.state_dict()['mlp_extractor.policy_net.0.bias'].detach().numpy()
    weights = model.policy.state_dict()['mlp_extractor.policy_net.0.weight'].detach().numpy()
    for i in range(len(biases)):
        w = weights[i]
        b = biases[i]
        relu_programs.append([w, b])
    pickle.dump(relu_programs, file=open(save_to + 'ReLUs.pkl', "wb"))
    return relu_programs


def main(seed, l1_actor, l2_actor):

    # configure directory
    save_to = './Oracle/' + str(l1_actor) + 'x' + str(l2_actor) + '/' + str(seed) + '/'
    if not os.path.exists(save_to):
        print(save_to)
        os.makedirs(save_to)

    if l2_actor == 0:
        net_arch = [l1_actor]
    else:
        net_arch = [l1_actor, l2_actor]

    # create environment
    seed = seed
    env = gym.make("CartPole-v1")
    env.seed(seed)

    # training completion requirements
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=500., verbose=0)
    eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, verbose=0)

    # train oracle
    model = PPO('MlpPolicy', env, seed=seed, verbose=0, policy_kwargs=dict(net_arch=[dict(pi=net_arch, vf=[128, 128])], activation_fn=torch.nn.ReLU))
    model.learn(int(1e10), callback=eval_callback)

    # save oracle
    model.save(save_to + 'model')
    model = model.load(save_to + 'model')

    # save ReLU programs from actor network
    save_relus(model, save_to)


if __name__ == "__main__":


    pool = multiprocessing.Pool(10)
    #pool.starmap(main, zip(range(1, 16), repeat(4), repeat(0)))
    #pool.starmap(main, zip(range(1, 31), repeat(32), repeat(0)))
    #pool.starmap(main, zip(range(11, 31), repeat(64), repeat(64)))
    pool.starmap(main, zip(range(1, 16), repeat(1), repeat(0)))
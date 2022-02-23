import gym
from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import torch
import os
import pickle
import multiprocessing
from itertools import repeat
import mujoco_py

"""

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

import mujoco_py
mj_path = mujoco_py.utils.discover_mujoco()
xml_path = os.path.join(mj_path, 'model', 'humanoid.xml')
model = mujoco_py.load_model_from_path(xml_path)
sim = mujoco_py.MjSim(model)

print(sim.data.qpos)
# [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

sim.step()
print(sim.data.qpos)




import gym
import mujoco_py
#Setting MountainCar-v0 as the environment
env = gym.make('Walker2d-v2')
#Sets an initial state
env.reset()

print(env.action_space.sample())
# Rendering our instance 300 times
for _ in range(300):
  #renders the environment
  env.render()
  #Takes a random action from its action space
  # aka the number of unique actions an agent can perform
  env.step(env.action_space.sample())
env.close()
"""




def save_relus(model, save_to):
    relu_programs = []
    biases = model.policy.state_dict()['mlp_extractor.policy_net.0.bias'].detach().numpy().tolist()
    weights = model.policy.state_dict()['mlp_extractor.policy_net.0.weight'].detach().numpy().tolist()
    for i in range(len(biases)):
        w = weights[i]
        b = biases[i]
        relu_programs.append([w, b])
    pickle.dump(relu_programs, file=open(save_to + 'ReLUs.pkl', "wb"))
    return relu_programs


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
    seed = seed
    env = gym.make('Walker2d-v3')
    env.seed(seed)

    # training completion requirements
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=4500., verbose=0)
    eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, verbose=2)

    # train oracle
    #model = DDPG('MlpPolicy', env, seed=seed, policy_kwargs=dict(net_arch=dict(qf=[256, 256], pi=[64, 64]), activation_fn=torch.nn.ReLU), verbose=0)

    #model = DDPG('MlpPolicy', env, seed=seed, policy_kwargs=dict(net_arch=[dict(pi=net_arch, vf=[256, 256])], activation_fn=torch.nn.ReLU))


    #model.learn(int(1e10), callback=eval_callback)

    # save oracle
    #model.save(save_to + 'model')

    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }


    model = PPO.load('./Oracle/256x256/0/HalfCheetah-v3', env=env, custom_objects=custom_objects)
    # save ReLU programs from actor network
    save_relus(model, save_to)




if __name__ == "__main__":

    main(00, 256, 256)
    #pool = multiprocessing.Pool(10)
    #pool.starmap(main, zip(range(1, 16), repeat(4), repeat(0)))
    #pool.starmap(main, zip(range(1, 16), repeat(32), repeat(0)))
    #pool.starmap(main, zip(range(1, 16), repeat(256), repeat(0)))
    #pool.starmap(main, zip(range(1, 16), repeat(64), repeat(64)))
    #pool.starmap(main, zip(range(6, 16), repeat(256), repeat(256)))



"""
def get_neurons(obs, relus):
    neurons = []
    for i, relu in enumerate(relus):
        neuron = max(0, np.dot(obs, relu[0]) + relu[1])
        neurons.append(neuron)
    neurons.extend(obs)
    return neurons


def my_program(obs, trees, relus):
    neurons = get_neurons(obs, relus)
    action = []
    for i, tree in enumerate(trees):
        action.append(tree.predict([neurons])[0])
    return action


def train_trees(x, y, depth, seed):
    trees = []

    for i in range(2):
        y_i = [item[i] for item in y]
        regr_tree = tree.DecisionTreeRegressor(max_depth=depth, random_state=seed)
        regr_tree.fit(x, y_i)
        trees.append(regr_tree)
    return trees




def augmented_dagger(env, model, save_to, depth, rollouts, eps_per_rollout, seed):

    X = np.load(save_to + "Neurons_" + str(eps_per_rollout) + ".npy").tolist()
    Y = np.load(save_to + "Actions_" + str(eps_per_rollout) + ".npy").tolist()
    relu_programs = pickle.load(open(save_to + "ReLUs.pkl", "rb"))
    best_reward = -10000

    for i in range(rollouts):
        # Regression tree
        trees = train_trees(X, Y, depth, seed)

        # Rollout
        steps = 0
        reward_avg = 0.
        for i in range(eps_per_rollout):
            ob = env.reset()
            reward = 0.0
            done = False
            while not done:
                # DAGGER
                X.append(get_neurons(ob, relu_programs))
                Y.append(model.predict(ob, deterministic=True)[0])
                # Interact with Environment
                action = my_program(ob, trees, relu_programs)
                ob, r_t, done, _ = env.step(action)
                steps += 1
                reward_avg += r_t

        # 100 consecutive eps
        for i in range(100 - eps_per_rollout):
            ob = env.reset()
            done = False
            while not done:
                action = my_program(ob, trees, relu_programs)
                ob, r_t, done, _ = env.step(action)
                reward_avg += r_t
        reward_avg = reward_avg / 100.

        # Update best program
        if reward_avg > best_reward:
            best_reward = reward_avg
            best_program = copy.deepcopy(trees)

    return best_reward, best_program


def base_dagger(env, model, save_to, depth, rollouts, eps_per_rollout, seed):

    X = np.load(save_to + "Observations_" + str(eps_per_rollout) + ".npy").tolist()
    Y = np.load(save_to + "Actions_" + str(eps_per_rollout) + ".npy").tolist()
    best_reward = -10000

    for i in range(rollouts):
        # Fit tree
        trees = train_trees(X, Y, depth, seed)

        # Rollout
        steps = 0
        reward_avg = 0.0
        for i in range(eps_per_rollout):
            ob = env.reset()
            done = False
            while not done:
                # DAGGER
                X.append(ob)
                Y.append(model.predict(ob, deterministic=True)[0])
                # Interact with Environment
                action = my_program(ob, trees, [])
                ob, r_t, done, _ = env.step(action)
                steps += 1
                reward_avg += r_t

        # 100 consecutive eps
        for i in range(100 - eps_per_rollout):
            ob = env.reset()
            done = False
            while not done:
                action = my_program(ob, trees, [])
                ob, r_t, done, _ = env.step(action)
                reward_avg += r_t
        reward_avg = reward_avg / 100.

        # Update best program
        if reward_avg > best_reward:
            best_reward = reward_avg
            best_program = copy.deepcopy(trees)

    return best_reward, best_program


    # generate experience
    #initialize_history(env, model, save_to, 25)

    # DAgger using (neuron, action) as training data. Various tree sizes.
    rewards = []
    programs = []
    for depth in range(1, 2): # range(1, 4):
        reward, program = augmented_dagger(env, model, save_to, depth, 25, 25, seed)
        rewards.append(reward)
        programs.append(program)
        print("Depth: ", depth)
        print("Reward: ", reward)
        print(tree.export_text(program[0]))
        print(tree.export_text(program[1]))

    np.save(file=save_to + 'AugTreeRewards1.npy', arr=rewards)
    pickle.dump(programs, file=open(save_to + 'AugTreePrograms1.pkl', "wb"))
   

    # DAgger using Base DSL.
    if l1_actor == 256 and l2_actor == 256 and True:

        if not os.path.exists('./Oracle/BaseDSL/' + str(seed)):
            os.makedirs('./Oracle/BaseDSL/' + str(seed))

        rewards = []
        programs = []
        for depth in range(1, 4):
            reward, program = base_dagger(env, model, save_to, depth, 25, 25, seed)
            rewards.append(reward)
            programs.append(program)
            print("Depth: ", depth)
            print("Reward: ", reward)

        np.save(file='./Oracle/BaseDSL/' + str(seed) + '/BaseTreeRewards1.npy', arr=rewards)
        pickle.dump(programs, file=open('./Oracle/BaseDSL/' + str(seed) + '/BaseTreePrograms1.pkl', "wb"))

def initialize_history(env, model, save_to, games):
    relu_programs = pickle.load(open(save_to + "ReLUs.pkl", "rb"))
    observations = []
    actions = []
    neurons = []
    r = 0.0

    for episode in range(games):
        state = env.reset()
        done = False
        while not done:
            action, _ = model.predict(state, deterministic=True)
            # Record Trajectory
            observations.append(state)
            actions.append(action)
            neurons.append(get_neurons(state, relu_programs))
            # Interact with Environment
            state, reward, done, _ = env.step(action)
            r += reward
    # Save Data
    np.save(file=save_to + 'Observations_' + str(games) + '.npy', arr=observations)
    np.save(file=save_to + 'Actions_' + str(games) + '.npy', arr=actions)
    np.save(file=save_to + 'Neurons_' + str(games) + '.npy', arr=neurons)

"""
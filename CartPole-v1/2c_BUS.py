import gym
from DSL_BUS import Ite, Lt, Observation, Num, AssignAction, Addition, Multiplication, Linear, ReLU
import numpy as np
import copy
from evaluation import DAgger
import pickle
import time
from stable_baselines3 import PPO
import os


def initialize_history(env, model, save_to, games):
    observations = []
    actions = []
    r = 0.0

    for episode in range(games):
        state = env.reset()
        done = False
        while not done:
            # Query oracle
            action = model.predict(state, deterministic=True)[0]
            # Record Trajectory
            observations.append(state)
            actions.append(action)
            # Interact with Environment
            state, reward, done, _ = env.step(action)
            r += reward
    r = r / games
    print("Oracle Reward:", r)
    np.savetxt(save_to + 'OracleReward.txt', [r])

    return observations, actions


def get_action(obs, p):
    actions = []
    for ob in obs:
        namespace = {'obs': ob, 'act': 0}
        p.interpret(namespace)
        actions.append(namespace['act'])
        # actions.append(namespace['act'].value)  # EDITED...
    return actions


class ProgramList():

    def __init__(self):
        self.plist = {}

    def insert(self, program):

        if program.getSize() not in self.plist:
            self.plist[program.getSize()] = {}

        if program.name() not in self.plist[program.getSize()]:
            self.plist[program.getSize()][program.name()] = []

        self.plist[program.getSize()][program.name()].append(program)

    def init_plist(self, constant_values, observation_values, action_values, linear_values, relu_programs):
        for i in observation_values:
            p = Observation(i)
            self.insert(p)

        for i in constant_values:
            p = Num(i)
            self.insert(p)

        for i in action_values:
            p = AssignAction(Num(i))
            self.insert(p)

        for i, w in enumerate(linear_values):
            p = Linear(w)
            self.insert(p)

        for i, relu in enumerate(relu_programs):
            p = ReLU(relu[0], relu[1])
            self.insert(p)

    def get_programs(self, size):

        if size in self.plist:
            return self.plist[size]
        return {}


class BottomUpSearch():

    def init_env(self, inout):
        env = {}
        for v in self._variables:
            env[v] = inout[v]
        return env

    def has_equivalent(self, program, observations, actions):
        p_out = get_action(obs=observations, p=program)
        tuple_out = tuple(p_out)

        if tuple_out not in self.outputs:
            self.outputs.add(tuple_out)
            return False
        return True

    def grow(self, plist, closed_list, operations, size):
        new_programs = []
        for op in operations:
            for p in op.grow(plist, size):
                if p not in closed_list:
                    closed_list.append(p)
                    new_programs.append(p)
                    yield p

        for p in new_programs:
            plist.insert(p)

    def imitate_oracle(self, bound, eval_fn, operations, constant_values, observation_values, action_values,
                       linear_values,
                       boolean_programs, PiRL=False):

        closed_list = []
        plist = ProgramList()
        plist.init_plist(constant_values, observation_values, action_values, linear_values, boolean_programs)

        best_score = 0.0
        best_policy = None
        number_evaluations = 0

        for current_size in range(2, bound + 1):
            for p in self.grow(plist, closed_list, operations, current_size):
                if p.name() == Ite.name():
                    p_copy = copy.deepcopy(p)
                    # Evaluate policy
                    if PiRL:
                        score = eval_fn.optimize(p_copy)
                    else:
                        score = eval_fn.evaluate(p_copy)
                    number_evaluations += 1
                    # Best imitation policy
                    if score > best_score:
                        best_policy = p_copy
                        best_score = score

        return best_policy, number_evaluations, best_score


def algo_NDPS(oracle_path, seed, roll_outs=25, eps_per_rollout=25, pomd='CartPole-v1'):
    # logs
    t0 = time.time()
    time_vs_reward = []

    # Task setup
    env = gym.make(pomd)
    np.random.seed(seed)
    env.seed(seed)
    best_reward = -10e7
    best_program = None

    # configure directory
    load_from = "./Oracle/" + oracle_path + '/' + str(seed) + '/'
    save_to = load_from + '2c/'
    if not os.path.exists(save_to):
        os.makedirs(save_to)

    # Specify DSL
    operations = [Ite, Lt]
    linear_values = [[0,0,0,0]]
    constant_values = [0.0]
    observation_values = []  # np.arange(env.observation_space.shape[0])
    action_values = np.arange(env.action_space.n)
    relu_programs = []
    max_size = 6

    # load oracle model, initialize dataset
    model = PPO.load(load_from + 'model')
    inputs, actions = initialize_history(env, model, save_to, eps_per_rollout)

    # Arguments for evaluation function
    oracle = {"oracle": model, "inputs": inputs, "actions": actions}
    synthesizer = BottomUpSearch()
    eval_fn = DAgger(oracle, nb_evaluations=100, seed=seed, env_name=pomd)

    # NDPS
    for r in range(roll_outs):
        # Imitation Step
        next_program, nb_evals, score = synthesizer.imitate_oracle(max_size, eval_fn, operations, constant_values,
                                                            observation_values, action_values,
                                                            linear_values, relu_programs, True)

        # Evaluate program
        reward = eval_fn.collect_reward(next_program, 100)
        print(reward)

        # Update program
        if reward > best_reward:
            best_reward = reward
            best_program = next_program

        # Update histories
        eval_fn.update_trajectory0(best_program, eps_per_rollout)

        # log results
        time_vs_reward.append([time.time() - t0, best_reward])
        print(r, best_reward, score)

    # save data
    np.save(file=save_to + 'Reward_1.npy', arr=reward)
    pickle.dump(best_program, file=open(save_to + 'Program_1.pkl', "wb"))
    np.save(file=save_to + 'TimeVsReward_1.npy', arr=time_vs_reward)

    # Display results
    print(save_to)
    print("Best Reward and Policy: ", best_reward)
    print(best_program.toString())


if __name__ == '__main__':
    for s in range(15, 16):
        #algo_NDPS("256x256", s)
        algo_NDPS("256x0", s)
        algo_NDPS("64x64", s)



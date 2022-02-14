import numpy as np
import gym
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

#    [x, y, v_x, v_y, theta, v_theta, c_L, c_R]
n0 = [-0.09,  0.12, -0.12,  0.27,  0.22,  0.18,  0.22,  0.1]
c0 = 0

def my_program(obs):
    obs = obs.tolist()
    if (np.dot(n0, obs) < c0):
        if 0.45 * obs[1] + obs[3] < -0.02:  # obs[1] + 2 * obs[3] < 0
            return 2  # fire main
        return 1  # fire left
    else:
        if obs[6] + obs[7] > 0:  # if either leg has contact
            return 0  # do nothing
        return 3  # fire right


env = gym.make("LunarLander-v2")
env.seed(999)
averaged = 0.0
games = 100

for i in range(games):
    ob = env.reset()
    reward = 0.0
    done = False
    while not done:
        env.render()
        action = my_program(ob)
        ob, r_t, done, _ = env.step(action)
        reward += r_t
    print(reward)
    averaged += reward
averaged /= games
print(games, averaged)


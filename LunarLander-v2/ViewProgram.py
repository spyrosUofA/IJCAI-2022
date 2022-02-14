import numpy as np
import gym

#    [x, y, v_x, v_y, theta, v_theta, c_L, c_R]
n0 = [-0.09,  0.12, -0.12,  0.27,  0.22,  0.18,  0.22,  0.1]
#n0 = [-0.0,  0.12, -0.12,  0.27,  0.22,  0.18,  0.22,  0.1]
n10 = [0.04,  0.08,  0.2,  0.2,  -0.24, -0.26,  0.21,  0.15]
#n10 = [0.04,  0.08,  0.2,  0.2,  -0.24, -0.26,  0, 0]  # simplified
#n10 = [0.04,  0.08,  0.2,  0.2,  -0.24, -0.26,  0, 0]  # simplified
n11 = [-0.04, -0.04,  0.19,  0.16, -0.13, -0.06,  0.29,  0.25]

def heuristic(env, s):
    """
    The heuristic for
    1. Testing
    2. Demonstration rollout.
    Args:
        env: The environment
        s (list): The state. Attributes:
                  s[0] is the horizontal coordinate
                  s[1] is the vertical coordinate
                  s[2] is the horizontal speed
                  s[3] is the vertical speed
                  s[4] is the angle
                  s[5] is the angular speed
                  s[6] 1 if first leg has contact, else 0
                  s[7] 1 if second leg has contact, else 0
    returns:
         a: The heuristic to be fed into the step function defined above to determine the next step and reward.
    """

    angle_targ = s[0] * 0.5 + s[2] * 1.0  # angle should point towards center
    if angle_targ > 0.4:
        angle_targ = 0.4  # more than 0.4 radians (22 degrees) is bad
    if angle_targ < -0.4:
        angle_targ = -0.4
    hover_targ = 0.55 * np.abs(
        s[0]
    )  # target y should be proportional to horizontal offset

    angle_todo = (angle_targ - s[4]) * 0.5 - (s[5]) * 1.0
    hover_todo = (hover_targ - s[1]) * 0.5 - (s[3]) * 0.5

    if s[6] or s[7]:  # legs have contact
        angle_todo = 0
        hover_todo = (
            -(s[3]) * 0.5
        )  # override to reduce fall speed, that's all we need after contact

    if env.continuous:
        a = np.array([hover_todo * 20 - 1, -angle_todo * 20])
        a = np.clip(a, -1, +1)
    else:
        a = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
            a = 2
        elif angle_todo < -0.05:
            a = 3
        elif angle_todo > +0.05:
            a = 1
    return a


def my_program0(obs):
    obs = obs.tolist()
    if np.dot(n0, obs) <= 0.0:
        if obs[3] < -0.02 + -0.45 * obs[1]: #np.dot(n10, obs) <= -0.01: # -0.01  ob[4] > 0.03: #
            return 2  # fire main
        return 1  # fire left
    else:
        if obs[6] + obs[7] > 0:  # if either leg has contact
            return 0  # do nothing
        return 3  # fire right


n0 = [-0.09,  0.12, -0.12,  0.27,  0.22,  0.18,  0.22,  0.1]

def my_program(obs):
    obs = obs.tolist()
    if np.dot(n0, obs) <= 0.0:
        if obs[3] + 0.45 * obs[1] < -0.02:
        #if obs[3] + 0.5 * obs[1] < -0.0: # 215.9
            return 2  # fire main
        return 1  # fire left
    else:
        if obs[6] + obs[7] > 0:  # if either leg has contact
            return 0  # do nothing
        return 3  # fire right


def my_program(obs):
    obs = obs.tolist()
    if np.dot(n0, obs) <= 0.0:
    #if obs[1] > 0.01 and obs[3] < 0 and obs[3] + 0.45 * obs[1] > -0.02:
        #if obs[3] + 0.5 * obs[1] < -0.0:
        #if obs[3] + 0.45 * obs[1] < -0.02:
        if np.dot(n10, obs) <= -0.01:
            return 2  # fire main
        return 1  # fire left
    else:
        if obs[6] + obs[7] > 0:  # if either leg has contact
            return 0  # do nothing
        return 3  # fire right




# Evaluate Policy
env = gym.make("LunarLander-v2")
env.seed(0)
averaged = 0.0
games = 100

x, y = [], []
x1, y1 = [], []
x2, y2 = [], []

for i in range(games):
    ob = env.reset()
    reward = 0.0
    done = False
    while not done:
        #env.render()
        action = my_program(ob)
        #action = heuristic(env, ob)


        x.append(ob)
        y.append(np.dot(ob, n0) <= 0.0)

        if 0 < action < 3:
           x1.append(ob)
           y1.append(action)
        else:
            x2.append(ob)
            y2.append(ob[6] + ob[7] > 0)

        ob, r_t, done, _ = env.step(action)
        reward += r_t
    print(reward)
    averaged += reward
averaged /= games
print(games, averaged)

np.savetxt("save_x.txt", x)
np.savetxt("save_y.txt", y)
np.savetxt("save_x1.txt", x1)
np.savetxt("save_y1.txt", y1)
np.savetxt("save_x2.txt", x2)
np.savetxt("save_y2.txt", y2)



from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

clf = make_pipeline(StandardScaler(),
                    LinearSVC(random_state=0, tol=1e-5, penalty='l1', loss='squared_hinge', dual=False))
clf.fit(x2, y2)


print(clf.named_steps['linearsvc'].coef_)

print(clf.named_steps['linearsvc'].intercept_)

print(clf.predict([ob]))



exit()



def my_program(obs):
    obs = obs.tolist()
    if np.dot(n0, obs) <= 0.0:
        if np.dot(n10, obs) <= -0.01:
            print(2, obs, "\n", 2, np.multiply(obs, n10).tolist())
            return 2  # fire main
        print(1, obs, "\n", 1,  np.multiply(obs, n10).tolist())
        return 1  # fire left
    else:
        if obs[6] + obs[7] > 0:  # if either leg has contact
            return 0  # do nothing
        return 3  # fire right


# Evaluate Policy
env = gym.make("LunarLander-v2")
env.seed(0)
averaged = 0.0
games = 1

for i in range(games):
    ob = env.reset()
    reward = 0.0
    done = False
    while not done:
        #env.render()
        action = my_program(ob)
        ob, r_t, done, _ = env.step(action)
        reward += r_t
    print(reward)
    averaged += reward
averaged /= games

print(games, averaged)


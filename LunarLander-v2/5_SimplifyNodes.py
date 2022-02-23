import numpy as np
import gym
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn import tree
import pickle
from sklearn.model_selection import GridSearchCV
from stable_baselines3 import PPO
from sklearn.pipeline import Pipeline


def relu_string(relus):
    names = []
    for _, relu in enumerate(relus):
        name = '(' + str(np.around(relu[0], 2)) + " *dot* obs[:] + " + str(np.round(relu[1], 2)) + ")"
        names.append(name)
    return names


def prepare_program(dec_tree, relus):
    used_features = [x for x in dec_tree.tree_.feature if 0 <= x < len(relus)] #list(set([x for x in dec_tree.tree_.feature if 0 <= x < len(relus)]))
    used_relus = []
    for j in used_features:
        used_relus.append(relus[j])
    return used_features, used_relus


def my_program0(obs, dec_tree, nb_relus, used_features, used_relus):
    all_features = [0] * nb_relus
    all_features.extend(obs)
    for i, relu in enumerate(used_relus):
        all_features[used_features[i]] = max(0, np.dot(relu[0], obs) + relu[1])
    return dec_tree.predict([all_features])[0]


# Specify Policy
oracle = "256x0"
method = "2a_NO_WEIGHT_FINAL"
depth = "2"
seed = "10"

oracle = "4x0"
method = "2a_VIPER_WEIGHT_FINAL"
depth = "2"
seed = "2"

# Load policy
policy = pickle.load(open("./Oracle/" + oracle + '/' + seed + '/' + method + '/Program_' + depth + '.pkl', "rb"))
relus = pickle.load(open("./Oracle/" + oracle + '/' + seed + "/ReLUs.pkl", "rb"))
relu_names = relu_string(relus)
relu_names.extend(["x", "y", "v_x", "v_y", "theta", "v_th", "c_L", "c_R"])
nb_relus = len(relus)
used_features, used_relus = prepare_program(policy, relus)
print(tree.export_text(policy, feature_names=relu_names))
model = PPO.load("./Oracle/" + oracle + '/' + seed + '/' + 'model')

# Extract decision rules (Depth First Search)
# Depth 1
n0 = used_relus[0][0]
c0 = policy.tree_.threshold[0] - used_relus[0][1]
# Depth 2
n1 = used_relus[1][0]
c1 = policy.tree_.threshold[1] - used_relus[1][1]
n2 = used_relus[2][0]
c2 = policy.tree_.threshold[4] - used_relus[2][1]

# Extract actions
a0 = np.argmax(policy.tree_.value[2][0])
a1 = np.argmax(policy.tree_.value[3][0])
a2 = np.argmax(policy.tree_.value[5][0])
a3 = np.argmax(policy.tree_.value[6][0])

# reconstruct tree
def my_program(obs):
    obs = obs.tolist()
    if np.dot(n0, obs) <= c0:
        if np.dot(n1, obs) <= c1:
            return a0
        return a1
    else:
        if np.dot(n2, obs) <= c2:
            return a2
        return a3


if False:
    step = 0
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

            # Generate Data
            # Split 0
            x.append(ob)
            #y.append(np.dot(ob, n0) <= c0)
            # Split 1
            if np.dot(ob, n0) <= c0:
                y.append(1)
                # Split L1
                x1.append(ob)
                if np.dot(n1, ob) <= c1:
                    y1.append(1)
                else:
                    y1.append(0)
            else:
                y.append(0)
                # Split R1
                x2.append(ob)
                if np.dot(n2, ob) <= c2:
                    y2.append(1)
                else:
                    y2.append(0)

            # Interact
            ob, r_t, done, _ = env.step(action)
            #print(action, model.predict(ob)[0])
            step += 1
            reward += r_t
        print(reward)
        averaged += reward
    averaged /= games
    print(games, averaged, step)

    np.savetxt("save_x.txt", x)
    np.savetxt("save_y.txt", y)
    np.savetxt("save_x1.txt", x1)
    np.savetxt("save_y1.txt", y1)
    np.savetxt("save_x2.txt", x2)
    np.savetxt("save_y2.txt", y2)

else:
    x = np.loadtxt("save_x.txt")
    y = np.loadtxt("save_y.txt")
    x1 = np.loadtxt("save_x1.txt")
    y1 = np.loadtxt("save_y1.txt")
    x2 = np.loadtxt("save_x2.txt")
    y2 = np.loadtxt("save_y2.txt")



# First Case Node
clf0 = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-4, penalty='l1', loss='squared_hinge', dual=False, C=0.005))


# Define a Standard Scaler to normalize inputs
scaler = StandardScaler()

# set the tolerance to a large value to make the example faster
linSVC = LinearSVC(random_state=0, tol=1e-4, penalty='l1', loss='squared_hinge', dual=False)
pipe = Pipeline(steps=[("scaler", scaler), ("linSVC", linSVC)])

# Parameters of pipelines can be set using ‘__’ separated parameter names:
param_grid = {"linSVC__C": np.logspace(-4, -3, 5)}
search = GridSearchCV(pipe, param_grid, n_jobs=1, verbose=3)
search.fit(x1, y1)



#print(search.cv_results_)
print(search.cv_results_['mean_test_score'])
##print(search.__dict__)
#print("Best parameter (CV score=%0.3f):" % search.best_score_)
##print(search.best_params_)
print(search.best_estimator_.named_steps['linSVC'].coef_)



clf0 = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-4, penalty='l1', loss='squared_hinge', dual=False, C=0.0005, max_iter=3000))
clf0.fit(x, y)
print(clf0.named_steps['linearsvc'].coef_, clf0.named_steps['linearsvc'].intercept_)
print(clf0.score(x, y))

# True Case Node
clf1 = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-4, penalty='l1', loss='squared_hinge', dual=False, C=0.05, max_iter=3000))
#clf1 = make_pipeline(LinearSVC(random_state=0, tol=1e-4, penalty='l1', loss='squared_hinge', dual=False, C=0.01))
clf1.fit(x1, y1) #[0:1000], y1[0:1000])
print(clf1.named_steps['linearsvc'].coef_, clf1.named_steps['linearsvc'].intercept_)
print(clf1.score(x1, y1))

#clf1.named_steps['linearsvc'].coef_[0] = [0, -0.45, 0, -1, 0, 0, 0, 0]
#clf1.named_steps['linearsvc'].intercept_[0] = -0.02
#print(clf1.score(x1, y1))


# False Case Node
clf2 = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-4, penalty='l1', loss='squared_hinge', dual=False, C=0.01, max_iter=3000))
clf2 = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-4, penalty='l1', loss='squared_hinge', dual=False, C=0.1, max_iter=3000))
#clf2 = make_pipeline(LinearSVC(random_state=0, tol=1e-1, penalty='l1', loss='squared_hinge', dual=False, C=0.01))
clf2.fit(x2, y2) #[0:2000], y2[0:2000])
print(clf2.named_steps['linearsvc'].coef_, clf2.named_steps['linearsvc'].intercept_)
print(clf2.score(x2, y2))


print(len(y), len(y1), len(y2))

#clf2.named_steps['linearsvc'].coef_[0] = [0, 0.5, 0, 1, 0, 0, 0, 0]
#clf2.named_steps['linearsvc'].coef_[0] = [0, 0, 0, 0, 0, 1, 0, 0]
#clf2.named_steps['linearsvc'].intercept_[0] = 0.0
#print(clf2.score(x1, y1))


# Simplified program
def new_program(obs):
    obs = [obs.tolist()]
    if clf0.predict(obs)[0]: # np.dot(n0, obs[0]) <= c0: #
        if clf1.predict(obs)[0]:
            return a0
        return a1
    else:
        if clf2.predict(obs)[0]:
            return a2
        return a3


# Test Program
env = gym.make("LunarLander-v2")
env.seed(1)
averaged = 0.0
games = 50
print()
for g in range(games):
    ob = env.reset()
    reward = 0.0
    done = False
    while not done:
        #env.render()
        action = new_program(ob)
        ob, r_t, done, _ = env.step(action)
        reward += r_t
    print(reward)
    averaged += reward
    env.close()
averaged /= games
print(games, averaged)

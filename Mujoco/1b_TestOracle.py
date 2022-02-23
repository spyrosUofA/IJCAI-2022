import gym
from stable_baselines3 import PPO
import numpy as np
import mujoco_py


def main(l1_actor, l2_actor):
    r_avg = []

    for seed in range(30, 31):
        # create environment
        env = gym.make('HalfCheetah-v3')
        env.seed(seed)

        # load oracle
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }

        model = PPO.load('./Oracle/256x256/0/HalfCheetah-v3', env=env, custom_objects=custom_objects)

        # run 100 episodes
        r = 0.0
        for episode in range(1):
            state = env.reset()
            print(state)

            done = False
            while not done:
                # Query oracle
                action = model.predict(state, deterministic=True)[0]
                # Interact with Environment
                state, reward, done, _ = env.step(action)
                r += reward
            print(r / (1+ episode))
        r_avg.append(r / 100)

    mean_Y = np.mean(r_avg, axis=0)
    std_Y = np.std(r_avg, axis=0)  # * (nb_seeds ** -0.5)
    print(mean_Y, std_Y)


if __name__ == "__main__":
    main(256, 256)


"""
[[ 2.0516992e+00 -2.9830152e-01 -1.4472986e-02 -4.8056450e-02
   2.2951923e-01  1.4106732e-02  6.8628401e-02 -1.6904305e-01
  -1.9363533e+00  8.5287884e-02  4.9057472e-03  2.9509078e-04
   2.4413221e-02 -4.5374180e-03  4.0177484e-03  2.7480276e-02
  -3.0743120e-02]]
"""
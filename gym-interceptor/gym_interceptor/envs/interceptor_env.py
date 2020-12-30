import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import tianshou
import gym
import Interceptor_V2
from predictor import MyInterceptor


class InterceptorEnv(gym.Env):
    def __init__(self):
        Interceptor_V2.Init()
        self.my_intr = MyInterceptor()
        self.stp = 0
        self.max_steps = 1000
        self.r_locs = []
        self.i_locs = []
        self.c_locs = []
        self.ang = 0
        self.game_score = 0
        self.my_score = 0
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(400, 31, 3))
        self.action_space = gym.spaces.Discrete(4)
        self.state = []
        self.score = 0
        self.done = False

    def reset(self):
        self.state = np.zeros(self.observation_space.shape)
        return self.state

    def step(self, action):
        if self.stp % 100 == 0:
            print("step", self.stp, "score", self.score, "rockets", len(self.r_locs))

        self.r_locs, self.i_locs, self.c_locs, self.ang, self.game_score = Interceptor_V2.Game_step(action)
        _, game_map, _ = self.my_intr.calculate_map_and_score(self.r_locs, self.i_locs, self.c_locs, self.ang, self.score, self.stp)
        self.my_score = self.my_intr.calc_score(action, game_map, self.ang)

        self.state = game_map
        reward = self.my_score

        self.stp += 1
        if self.stp >= self.max_steps:
            self.done = True
        else:
            self.done = False
        info = []
        info = {"episode" : None, "is_success" : None}
        return self.state, reward, self.done, info

    def seed(self, s):
        result = []
        return result

    def render(self, mode='rgb_array'):
        if mode == 'human':
            game_map = self.state
            im = game_map.astype("uint8")
            max_time = 400
            angs_options = 31
            im = cv.resize(im, (angs_options * 40, max_time * 3), interpolation=cv.INTER_NEAREST)
            im = 255 - im
            im = im.astype("uint8")
            im = np.vstack((cv.resize(im[:1, :], (angs_options * 40, 10)), im))
            im = im[::-1]
            im = 255-im
            return im
        elif mode == 'rgb_array':
            return self.state

    def close(self):
        return None


def main():
    a = 1


if __name__ == '__main__':
    main()

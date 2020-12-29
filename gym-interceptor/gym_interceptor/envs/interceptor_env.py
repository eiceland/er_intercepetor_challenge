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

        _, game_map, _ = self.my_intr.calculate_map_and_score(self.r_locs, self.i_locs, self.c_locs, self.ang, self.score, self.stp)
        self.my_score = self.my_intr.calc_score(action, game_map, self.ang)
        self.r_locs, self.i_locs, self.c_locs, self.ang, self.game_score = Interceptor_V2.Game_step(action)

        self.state = game_map
        reward = self.my_score

        self.done = False
        info = []
        self.stp += 1
        info = {"episode" : None, "is_success" : None}
        return self.state, reward, self.done, info

    def seed(self, s):
        result = []
        return result

    def render(self, mode):
        res_map = []
        return res_map

    def close(self):
        return None


def main():
    a = 1


if __name__ == '__main__':
    main()

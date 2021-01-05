import numpy as np
import cv2 as cv
import gym
import Interceptor_V2
from predictor import MyInterceptor


class InterceptorEnv(gym.Env):
    def __init__(self):
        Interceptor_V2.Init()
        self.my_intr = MyInterceptor()
        self.max_steps = 1000
        self.stp = 0
        self.r_locs = []
        self.i_locs = []
        self.c_locs = []
        self.ang = 0
        self.game_score = 0
        self.my_score = 0
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(401, 31, 3))
        self.action_space = gym.spaces.Discrete(4)
        self.state = None
        self.done = False
        self.reward_factor = 0.68

    def reset(self):
        self.stp = 0
        self.r_locs = []
        self.i_locs = []
        self.c_locs = []
        self.ang = 0
        self.game_score = 0
        self.my_score = 0

        self.my_intr.__init__()
        Interceptor_V2.Init()
        self.state = self.my_intr.calculate_map(self.r_locs, self.c_locs, self.ang, self.stp)
        return self.state

    def step(self, action):

        if action == self.my_intr.SHOOT:
            # Here we delete the interception points from the game map
            reward = self.my_intr.shoot()
        else:
            reward = 0
        self.my_score += reward

        if self.stp % 100 == 0:
            print("step", self.stp, "score", reward, "total score", self.my_score, "game score", self.game_score, "rockets", len(self.r_locs))
        self.stp += 1
        #next round: see what's new in the world
        self.r_locs, self.i_locs, self.c_locs, self.ang, self.game_score  = Interceptor_V2.Game_step(action)
        #now calcluate the game map for the next round
        game_map = self.my_intr.calculate_map(self.r_locs, self.c_locs, self.ang, self.stp)

        self.state = game_map
        if self.stp >= self.max_steps:
            self.done = True
            print("step", self.stp, "score", reward, "total score", self.my_score, "rockets", len(self.r_locs),
                  "game score", self.game_score)
        else:
            self.done = False
        info = {
            "episode": None,
            "is_success": True,
            "rockets": len(self.r_locs),
            "total score": self.my_score,
            "game score": self.game_score,
            "score": reward
        }
        adjusted_reward = reward * self.reward_factor
        return self.state, adjusted_reward, self.done, info

    def seed(self, s):
        result = []
        return result

    def render(self, mode='rgb_array'):
        if mode == 'human':
            game_map = self.state
            im = game_map.astype("uint8")
            max_time = im.shape[0]
            angs_options = im.shape[1]
            im_up = im[:angs_options, :, :]
            im_dwn = im[angs_options:, :, :]
            im_up = cv.resize(im_up, (angs_options * 40, max_time), interpolation=cv.INTER_NEAREST)
            im_dwn = cv.resize(im_dwn, (angs_options * 40, max_time), interpolation=cv.INTER_NEAREST)
            im = np.vstack((im_up, im_dwn))
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

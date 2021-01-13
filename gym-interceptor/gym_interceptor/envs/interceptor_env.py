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
        #self.observation_space = gym.spaces.Box(low=0, high=255, shape=(401, 31, 3))
        # self.observation_space = gym.spaces.Box(low=0, high=255, shape=(100, 31,1), dtype = np.uint8)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(100, 61, 1), dtype = np.uint8)
        self.action_space = gym.spaces.Discrete(4)
        self.state = None
        self.done = False
        self.reward_factor = 1#0.68
        self.city_rockets =0
        self.city_hits = 0
        self.city_inter = 0
        self.empty_inter = 0
        self.f_rockets = 0
        self.f_hits = 0
        self.f_inter = 0


    def reset(self):
        self.stp = 0
        self.r_locs = []
        self.i_locs = []
        self.c_locs = []
        self.ang = 0
        self.game_score = 0
        self.my_score = 0
        self.city_rockets =0
        self.city_hits = 0
        self.city_inter = 0
        self.empty_inter = 0
        self.f_rockets = 0
        self.f_hits = 0
        self.f_inter = 0

        self.my_intr.__init__()
        Interceptor_V2.Init()
        self.state,_,_,_,_ = self.my_intr.calculate_map(self.r_locs, self.c_locs, self.ang, self.stp)
        new_game_map = np.zeros((401, 61))
        player_loc = np.argmax(self.state[0])
        shift = 30 - player_loc
        new_game_map[:, shift:shift+31] = self.state
        self.state = new_game_map
        return np.expand_dims(self.state[:100], axis=2)

    def step(self, action):
        city_inter = 0
        f_inter = 0
        reward = 0
        if action == self.my_intr.SHOOT:
            # Here we delete the interception points from the game map
            reward, city_inter, f_inter, empty_shoot = self.my_intr.shoot()
            if empty_shoot:
                reward -= 1
                self.empty_inter += 1

        self.my_score += reward
        self.city_inter += city_inter
        self.f_inter += f_inter
        # if self.stp % 100 == 0:
            # print("step", self.stp, "score", reward, "total score", self.my_score, "rockets", len(self.r_locs),
            #       "game score", self.game_score, "city rockets", self.city_rockets, "city hits", self.city_hits,
            #       "city interceptions", self.city_inter, "field rockets", self.f_rockets, "field hits", self.f_hits,
            #       "field interceptions", self.f_inter, "empty shoots", self.empty_inter)
        self.stp += 1
        #next round: see what's new in the world
        self.r_locs, self.i_locs, self.c_locs, self.ang, self.game_score  = Interceptor_V2.Game_step(action)
        #now calcluate the game map for the next round
        game_map, new_city_rocket, n_city_hits, new_f_rocket, n_f_hits = self.my_intr.calculate_map(self.r_locs, self.c_locs, self.ang, self.stp)
        new_game_map = np.zeros((401, 61))
        player_loc = np.argmax(game_map[0])
        shift = 30 - player_loc
        new_game_map[:, shift:shift+31] = game_map

        self.city_rockets += new_city_rocket
        self.city_hits += n_city_hits
        self.f_rockets += new_f_rocket
        self.f_hits += n_f_hits

        self.state = new_game_map
        if self.stp >= self.max_steps:
            self.done = True
            print("game score", self.game_score, "step", self.stp, "score", reward, "total score", self.my_score, "rockets", len(self.r_locs),
                  "city rockets", self.city_rockets, "city hits", self.city_hits,
                  "city interceptions", self.city_inter, "field rockets", self.f_rockets, "field hits", self.f_hits,
                  "field interceptions", self.f_inter, "empty shoots", self.empty_inter, "\n")
        else:
            self.done = False
        info = {
            "episode": None,
            "is_success": True,
            "rockets": len(self.r_locs),
            "total score": self.my_score,
            "game score": self.game_score,
            "score": reward,
            "city rockets": self.city_rockets,
            "city hits": self.city_hits,
            "city interceptions": self.city_inter,
            "field rockets": self.f_rockets,
            "field hits": self.f_hits,
            "field interceptions": self.f_inter,
            "empty shoots": self.empty_inter
        }
        adjusted_reward = reward * self.reward_factor
        return np.expand_dims(self.state[:100], axis=2), adjusted_reward, self.done, info

    def seed(self, s):
        result = []
        return result

    def render(self, mode='rgb_array'):
        if mode == 'human':
            game_map = self.state
            im = game_map.astype("uint8")
            max_time = im.shape[0]
            angs_options = im.shape[1]
            interest_zone = 100
            im_up = im[:interest_zone, :]
            im_dwn = im[interest_zone:, :]
            len = 1600#max_time
            im_up = cv.resize(im_up, (len, angs_options * 40), interpolation=cv.INTER_NEAREST)
            im_dwn = cv.resize(im_dwn, (len, angs_options * 40), interpolation=cv.INTER_NEAREST)
            im = np.vstack((im_up, im_dwn))
            im = 255 - im
            im = im.astype("uint8")
            im = np.vstack((cv.resize(im[:1, :], (len, 10)), im))
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

import numpy as np
from Interceptor_V2 import Init, Draw, Game_step
import game_map
import itertools
import matplotlib.pyplot as plt


##Enums:
LEFT = 0
STRAIGHT = 1
RIGHT = 2
SHOOT = 3


class MyRocket:
    id_iter = itertools.count()

    def __init__(self, x1, y1, cur_time, city_list, intr_paths_mat):
        self.id = next(self.id_iter)
        self.path = []
        self.city_hit = False
        self.interception_points = []  # each interception point holds: inteceptor shooting time, angle, expected interception time
        self.city_list = city_list
        self.x0 = 4800.0
        self.y0 = 0.0
        self.x1 = x1
        self.y1 = y1
        self.shoot_time = cur_time - 2 #r_locs represents the situation in time cur_time-1.
        self.intr_paths_mat = intr_paths_mat
        self.index_2_ang = {i: -90 + 6 * i for i in range(31)}

        self.update_rocket_path_and_city_hit()
        self.calculate_interception_points()

    def update_rocket_path_and_city_hit(self, fric=5e-7, dt=0.2, g=9.8):
        dx, dy = self.x1 - self.x0, self.y1 - self.y0
        vx, vy = dx / dt, dy / dt
        self.path.append([self.x0, self.y0, self.shoot_time])
        x, y, t = self.x1, self.y1, self.shoot_time + 1
        while y >= 0:
            self.path.append([x, y, t])
            t += 1
            v_loss = (vx ** 2 + vy ** 2) * fric * dt
            vx = vx * (1 - v_loss)
            vy = vy * (1 - v_loss) - g * dt
            x = x + vx * dt
            y = y + vy * dt

        for c in self.city_list:
            if np.abs(x - c[0]) < c[1]:  # c[0] = x coordinate of city center, c[1] = distance from city center to the edge
                self.city_hit = True

    def calculate_interception_points(self, prox_radius=150):
        for t in range(2, len(self.path)): #the first two locations are in the past.
            r_path_t = np.array(self.path[t:])
            min_len = min(self.intr_paths_mat.shape[1], r_path_t.shape[0])
            squared_dif = (self.intr_paths_mat[:, :min_len, :] - r_path_t[:min_len, :2]) ** 2
            dists_squared = squared_dif[:, :, 0] + squared_dif[:, :, 1]
            meeting_points = np.nonzero(dists_squared < prox_radius ** 2)
            last_index = -1
            for index, time in zip(*meeting_points):
                if last_index == index:
                    continue
                last_index = index
                self.interception_points.append(
                    [self.shoot_time + t, self.index_2_ang[index], self.shoot_time + t + time])


class MyInterceptor:
    def __init__(self):
        self.rockets_list = []
        self.strategy = None
        self.time = 0
        self.t_no_new_rocket = 0
        self.intr_paths_mat = self.get_interceprtion_mat()
        self.game_map = game_map.GameMap()
        self.new_rockets = []
        self.removed_rockets = []

    def get_interceptor_path(self, ang0, v0=800, x0=-2000.0, y0=0.0, world_width=10000, fric=5e-7, dt=0.2, g=9.8):
        max_path_len = 353
        vx = v0 * np.sin(np.deg2rad(ang0))
        vy = v0 * np.cos(np.deg2rad(ang0))
        path = []
        path.append([x0, y0])
        x, y = x0, y0
        while (y >= 0) and (np.abs(x) <= world_width / 2):
            v_loss = (vx ** 2 + vy ** 2) * fric * dt
            vx = vx * (1 - v_loss)
            vy = vy * (1 - v_loss) - g * dt
            x = x + vx * dt
            y = y + vy * dt
            path.append([x, y])
        while (len(path) < max_path_len + 1):
            path.append([10000.0, 0.0])
        return path[:max_path_len]

    def get_interceprtion_mat(self):
        paths = []
        for ang in range(-90, 91, 6):
            paths.append(self.get_interceptor_path(ang))
        return np.array(paths)

    def is_new_rocket(self, x1, y1):
        if ((x1 - 4800.0)**2 + y1 **2) > 400**2:
            return False
        if self.t_no_new_rocket > 3:
            return True
        if (len(self.rockets_list) > 0):
            expected_loc = (self.rockets_list[-1]).path[self.t_no_new_rocket + 1][:2]
            if (np.abs(np.array(expected_loc) - np.array([x1, y1])) < 1e-6).all():
                return False
        return True

    def calculate_map_and_score(self, r_locs, i_locs, c_locs, ang, score, t):
        self.time = t
        self.t_no_new_rocket += 1
        self.new_rockets = []

        if (len(r_locs) > 0):
            x1, y1 = r_locs[-1]
            if (self.is_new_rocket(x1, y1)):
                rocket = MyRocket(*r_locs[-1], t, c_locs, self.intr_paths_mat)
                self.rockets_list.append(rocket)
                self.t_no_new_rocket = 0
                self.new_rockets.append(rocket.id)

        cur_game_map1 = self.game_map.update_map(self.time, self.rockets_list, self.new_rockets, self.removed_rockets, ang)
        compare = False#True#
        if compare:
            cur_game_map = game_map.build_game_map(self.rockets_list, ang, self.time)  ##Rafi
            if not (cur_game_map == cur_game_map1).all():
                plt.imshow(np.hstack((cur_game_map, 255*np.ones((cur_game_map.shape[0], 1, 3)), cur_game_map1, 255*np.ones((cur_game_map.shape[0], 1, 3)), np.abs(cur_game_map1-cur_game_map))))
                plt.show()
            else:
                print("assertion pass")

        self.removed_rockets = []
        if (self.strategy is None) and (len(self.rockets_list) > 0):
            rocket_to_hit = self.rockets_list[0]
            for p in rocket_to_hit.interception_points:
                self.strategy = p
                if ((p[0] - t) * 6 > np.abs(p[1] - ang)):  # p[0] absolute shooting time. p[1] angle to shoot
                    break
                self.strategy = None

        if self.strategy is None:
            if (ang > 45):
                action =  RIGHT
            else:
                action = LEFT
        else:
            if self.strategy[1] > ang:
                action = RIGHT
            elif self.strategy[1] < ang:
                action = LEFT
            elif self.strategy[0] == t:
                #action = SHOOT
                #self.strategy = None
                #self.removed_rockets.append(self.rockets_list[0].id)
                #self.rockets_list.remove(self.rockets_list[0])
                action = STRAIGHT
            else:
                action = STRAIGHT

        # remove rockets that landed
        for r in self.rockets_list:
            if r.path[-1][2] < t:
                self.removed_rockets.append(r.id)
                self.rockets_list.remove(r)
        return action, cur_game_map1, 0

    def calc_score(self, action, game_map, ang):
        #TODO: calc real scores
        self.my_score = 0
        if action == SHOOT:
            if game_map[0, game_map.ang2coord(ang), :].max() > 32: ## a city rocket
                self.my_score = 14
            elif game_map[0, game_map.ang2coord(ang), :].max() > 0: ## a field rocket
                self.my_score = 4
            else:
                self.my_score = 0

        #TODO: what about double-interceptions? Do they get double-score?
        return self.my_score


def my_main():
    Init()
    my_intr = MyInterceptor()
    r_locs = i_locs = c_locs = []
    ang = score = 0
    for stp in range(1000):
        if stp % 100 == 0:
            print("step", stp, "score", score, "rockets", len(r_locs))
        action_button, game_map, my_score = my_intr.calculate_map_and_score(r_locs, i_locs, c_locs, ang, score, stp)
        r_locs, i_locs, c_locs, ang, score = Game_step(action_button)
    # Draw()
    print(score)


import cProfile

if __name__ == '__main__':
    #cProfile.run('my_main()')
    my_main()

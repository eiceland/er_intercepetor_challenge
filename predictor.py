import numpy as np
import game_map
import itertools


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
        self.hit_time = 0
        self.end_game_time = 1000

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
        self.hit_time = self.path[-1][2]
        if self.hit_time > self.end_game_time:
            self.city_hit = False

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
    ##Enums:
    LEFT = 0
    STRAIGHT = 1
    RIGHT = 2
    SHOOT = 3

    def __init__(self):
        self.rockets_list = []
        self.strategy = None
        self.time = 0
        self.t_no_new_rocket = 0
        self.intr_paths_mat = self.get_interceprtion_mat()
        self.shoot_interval = 8
        self.game_map = game_map.GameMap(self.shoot_interval)
        self.last_shoot_time = -self.shoot_interval

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

    def calculate_map(self, r_locs, c_locs, ang, t):
        self.time = t
        self.t_no_new_rocket += 1
        new_rockets = []
        new_city_rocket = 0
        new_f_rocket = 0
        if (len(r_locs) > 0):
            x1, y1 = r_locs[-1]
            if (self.is_new_rocket(x1, y1)):
                rocket = MyRocket(*r_locs[-1], t, c_locs, self.intr_paths_mat)
                self.rockets_list.append(rocket)
                self.t_no_new_rocket = 0
                new_rockets.append(rocket.id)
                if rocket.city_hit:
                    new_city_rocket += 1
                else:
                    new_f_rocket += 1

        time_since_shoot = self.time - self.last_shoot_time
        cur_game_map = self.game_map.update_map(self.time, self.rockets_list, new_rockets, ang, time_since_shoot)
        city_hits = 0
        f_hits = 0
        for r in self.rockets_list:
            if r.path[-1][2] < t:
                if r.city_hit:
                    city_hits += 1
                else:
                    f_hits += 1
                self.rockets_list.remove(r)
        return cur_game_map, new_city_rocket, city_hits, new_f_rocket, f_hits

    def shoot(self):
        score = 0
        city_inter = 0
        f_inter = 0
        if self.time - self.last_shoot_time < self.shoot_interval:
            return -1, 0, 0, False
        self.last_shoot_time = self.time
        removed_rockets = self.game_map.delete_rockets_path_after_shoot(self.rockets_list)
        for id in removed_rockets:
            id_score, id_city = self.remove_rocket_from_list_and_get_score(id)
            score += id_score
            city_inter += id_city
            f_inter += (1 - id_city)
        if score > 18:
            print("interception of {} points".format(score))
        return score, city_inter, f_inter, score <= 0

    def remove_rocket_from_list_and_get_score(self, id):
        r_to_remove = None
        for r in self.rockets_list:
            if r.id == id:
                r_to_remove = r
                break
        score = 18 if r_to_remove.city_hit else 4
        self.rockets_list.remove(r_to_remove)
        return score, 1 if r_to_remove.city_hit else 0

    def calc_score(self, action, cur_game_map, ang):
        self.my_score = 0
        if action == self.SHOOT:
            if cur_game_map[1, game_map.ang2coord(ang), :].max() > 32: ## a city rocket
                self.my_score = 14
            elif cur_game_map[1, game_map.ang2coord(ang), :].max() > 0: ## a field rocket
                self.my_score = 4
            else:
                self.my_score = 0
        return self.my_score

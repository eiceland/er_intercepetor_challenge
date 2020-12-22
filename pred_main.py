import numpy as np
from Interceptor_V2 import Init, Draw, Game_step
import game_map

##Enums:
LEFT = 0
STRAIGHT = 1
RIGHT = 2
SHOOT = 3
#
# #given two succesive points in a path, compute the next n points.
# def get_path(x0, y0, x1, y1, n, fric = 5e-7, dt = 0.2 , g = 9.8):
#     dx, dy = x1 - x0, y1 - y0
#     vx, vy = dx / dt, dy / dt
#     x, y = x1, y1
#     path = []
#     for _ in range(n):
#         v_loss = (vx ** 2 + vy ** 2) * fric * dt
#         vx = vx * (1 - v_loss)
#         vy = vy * (1 - v_loss) - g * dt
#         x = x + vx * dt
#         y = y + vy * dt
#         path.append((x,y))
#     return path

# #given two succesive points in a path, compute the rest of the path.
# def get_rocket_full_path_and_city_hit(x0, y0, x1, y1, city_list, t, fric = 5e-7, dt = 0.2 , g = 9.8):
#     dx, dy = x1 - x0, y1 - y0
#     vx, vy = dx / dt, dy / dt
#     x, y = x1, y1
#     path = []
#     path.append([x0, y0, t])
#     t += 1
#     path.append([x1, y1, t])
#     while y >= 0:
#         t += 1
#         v_loss = (vx ** 2 + vy ** 2) * fric * dt
#         vx = vx * (1 - v_loss)
#         vy = vy * (1 - v_loss) - g * dt
#         x = x + vx * dt
#         y = y + vy * dt
#         path.append([x, y, t])
#
#     city_hit = False
#     for c in city_list:
#         if np.abs(x - c[0]) < c[1]: # c[0] = x coordinate of city center, c[1] = distance from city center to the edge
#             city_hit = True
#     return path, city_hit
#
# def get_interceptor_full_path(ang0, t, v0 = 800, x0 = -2000, y0 = 0, world_width = 10000, fric = 5e-7, dt = 0.2 , g = 9.8):
#
#     vx = v0 * np.sin(np.deg2rad(ang0))
#     vy = v0 * np.cos(np.deg2rad(ang0))
#     path = []
#     path.append([x0, y0, t])
#     x, y = x0, y0
#     while (y >= 0) and (np.abs(x) <= world_width / 2):
#         t += 1
#         v_loss = (vx ** 2 + vy ** 2) * fric * dt
#         vx = vx * (1 - v_loss)
#         vy = vy * (1 - v_loss) - g * dt
#         x = x + vx * dt
#         y = y + vy * dt
#         path.append([x, y, t])
#     return path
#
#
# def get_interception_times_and_angles(ang0, rocket_path, intr_path_dict, prox_radius = 150):
#
#     times_and_angles = []
#     for t in range(len(rocket_path)):
#         for ang in range(-84, 90, 6):
#             intr_path = intr_path_dict[ang]
#             if np.abs(ang - ang0) < 6*t:
#                 r_path_t = np.array(rocket_path[t:])
#                 min_len = min(intr_path.shape[0], r_path_t.shape[0])
#                 squared_dif = (intr_path[:min_len, :2] - r_path_t[:min_len, :2])**2
#                 dists_squared = squared_dif[:, 0] + squared_dif[:, 1]
#                 if (dists_squared < prox_radius**2).any():
#                     #Q: why argmin (closest point) and not first hitting point?
#                     inter_range = np.argmin(dists_squared)
#                     #Q: Maybe we want to have the total time till interception and not time from shooting.
#                     times_and_angles.append([t, ang, inter_range])
#
#     return times_and_angles

#
# def check_path_n_steps(n):
#     Init()
#     path_computed = False
#     computed_path = []
#     actual_path = []
#     while len(actual_path) < n:
#         x0 = 4800
#         y0 = 0
#         action_button = STRAIGHT
#         r_locs, i_locs, c_locs, ang, score = Game_step(action_button)
#         if path_computed:
#             actual_path.append(r_locs[0])
#         if r_locs.shape[0] > 0 and not path_computed:
#             x1, y1 = r_locs[-1]
#             computed_path = get_path(x0, y0, x1, y1, n)
#             path_computed = True
#     assert (np.abs((np.array(computed_path)[:n] - np.array(actual_path)[:n])) < 0.0000001).all()
#     print ("path computed successfully for {} steps".format(n))

# def check_full_path_and_city_hit():
#     Init()
#     x0 = 4800
#     y0 = 0
#     path_computed = False
#     computed_path = []
#     actual_path = []
#     score = 0
#
#     actual_path.append([x0,y0])
#     while (path_computed == False) or (len(actual_path) < len(computed_path)):
#         action_button = STRAIGHT
#         r_locs, i_locs, c_locs, ang, score = Game_step(action_button)
#         if r_locs.shape[0] > 0:
#             actual_path.append(r_locs[0])
#             if not path_computed:
#                 x1, y1 = r_locs[-1]
#                 computed_path, computed_hit = get_rocket_full_path_and_city_hit(x0, y0, x1, y1, c_locs)
#                 path_computed = True
#     assert (np.abs( np.array(computed_path[:-1]) - np.array(actual_path[:-1]) ) < 0.0000001).all()
#     print("rocket path computed successfully till hit")
#     assert (score <= -14*computed_hit - 1)
#     print ("city hit: {} was computed successfully".format(computed_hit))
#

# def check_interceptor_path():
#     Init()
#     path_computed = False
#     computed_path = []
#     actual_path = []
#     while (path_computed == False) or (len(actual_path) < len(computed_path)):
#         action_button = np.random.randint(0,4)
#         r_locs, i_locs, c_locs, ang, score = Game_step(action_button)
#         if i_locs.shape[0] > 0:
#             actual_path.append(i_locs[0])
#             if not path_computed:
#                 computed_path = get_interceptor_full_path(ang)
#                 path_computed = True
#     assert (np.abs(np.array(computed_path[:-1]) - np.array(actual_path[:-1])) < 0.0000001).all()
#     print("interceptor path of length {} computed successfully".format(len(computed_path[:-1])))
# #
# def check_interception_points():
#     Init()
#     x0 = 4800
#     y0 = 0
#     path_computed = False
#     rocket_path = []
#     intr_path_dict = {ang: np.array(get_interceptor_full_path(ang)[:-1]) for ang in range(-84, 90, 6)}
#     while (path_computed == False):
#         action_button = STRAIGHT
#         r_locs, i_locs, c_locs, ang, score = Game_step(action_button)
#         if r_locs.shape[0] > 0:
#             x1, y1 = r_locs[-1]
#             rocket_path, computed_hit = get_rocket_full_path_and_city_hit(x0, y0, x1, y1, c_locs, 0)
#             path_computed = True
#     t_and_ang = get_interception_times_and_angles (ang, rocket_path, intr_path_dict)
#     print (rocket_path)
#     print(t_and_ang)

# def my_unitests():
    # check_path_n_steps(10)
    # check_full_path_and_city_hit()
    # this function might fail if the interceptor did not complete its path (since it met a rocket...)
    # check_interceptor_path()
    # check_interception_points()

class MyRocket():
    def __init__(self, x1, y1, cur_time, city_list, intr_path_dict):
        self.path = []
        self.city_hit = False
        self.interception_points = [] #each interception point holds: inteceptor shooting time, angle, expected interception time
        self.city_list = city_list
        self.x0 = 4800.0
        self.y0 = 0.0
        self.x1 = x1
        self.y1 = y1
        self.shoot_time = cur_time
        self.intr_path_dict = intr_path_dict
        self.update_rocket_path_and_city_hit()
        self.calculate_interception_points()

    def update_rocket_path_and_city_hit(self, fric = 5e-7, dt = 0.2 , g = 9.8):
        dx, dy = self.x1 - self.x0, self.y1 - self.y0
        vx, vy = dx / dt, dy / dt
        self.path.append([self.x0, self.y0, self.shoot_time ])
        x, y, t = self.x1, self.y1, self.shoot_time + 1
        self.path.append([x, y, t])
        while y >= 0:
            t += 1
            v_loss = (vx ** 2 + vy ** 2) * fric * dt
            vx = vx * (1 - v_loss)
            vy = vy * (1 - v_loss) - g * dt
            x = x + vx * dt
            y = y + vy * dt
            self.path.append([x, y, t])

        for c in self.city_list:
            if np.abs(x - c[0]) < c[1]: # c[0] = x coordinate of city center, c[1] = distance from city center to the edge
                self.city_hit = True

    def calculate_interception_points(self, prox_radius = 150):
        for t in range(len(self.path)):
            for ang in range(-90, 91, 6):
                intr_path = self.intr_path_dict[ang]
                r_path_t = np.array(self.path[t:])
                min_len = min(intr_path.shape[0], r_path_t.shape[0])
                squared_dif = (intr_path[:min_len, :2] - r_path_t[:min_len, :2]) ** 2
                dists_squared = squared_dif[:, 0] + squared_dif[:, 1]
                if (dists_squared < prox_radius ** 2).any():
                    intr_time = np.argmin(dists_squared)
                    self.interception_points.append([self.shoot_time+t, ang, self.shoot_time + t + intr_time])

class MyInterceptor():
    def __init__(self):
        self.r_city_list = []
        self.r_field_list = []
        self.strategy = None
        self.time = 0
        self.intr_path_dict = {ang: np.array(self.get_interceptor_path(ang)[:-1]) for ang in range(-90, 91, 6)}
        self.t_no_new_city_r = 0
        self.t_no_new_field_r = 0

    def get_interceptor_path(self, ang0, v0 = 800, x0 = -2000, y0 = 0, world_width = 10000, fric = 5e-7, dt = 0.2 , g = 9.8):
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
        return path

    def is_new_rocket(self, x1,y1):
        if (len(self.r_city_list) > 0):
            expected_loc =  (self.r_city_list[-1]).path[self.t_no_new_city_r+1][:2]
            if (np.abs(np.array(expected_loc) - np.array([x1,y1])) < 1e-6).all():
                return False
        if (len(self.r_field_list) > 0):
            expected_loc =  (self.r_field_list[-1]).path[self.t_no_new_field_r+1][:2]
            if (np.abs(np.array(expected_loc) - np.array([x1, y1])) < 1e-6).all():
                return False
        return True

    def calculate_action(self, r_locs, i_locs, c_locs, ang, score, t):

        self.t_no_new_city_r += 1
        self.t_no_new_field_r += 1
        if (len(r_locs)>0):
            x1,y1 = r_locs[-1]
        if (len(r_locs)>0) and (self.is_new_rocket(x1,y1)):
            rocket = MyRocket(*r_locs[-1], t, c_locs, self.intr_path_dict )
            if rocket.city_hit:
                self.r_city_list.append(rocket)
                self.t_no_new_city_r = 0
            else:
                self.r_field_list.append(rocket)
                self.t_no_new_field_r = 0

        #cur_game_map = game_map.build_game_map(self.t_and_ang_city_all, self.t_and_ang_field_all, ang)  ##Rafi

        while (self.strategy is None) and (len(self.r_city_list) > 0):
            rocket_to_hit = self.r_city_list[0]
            for p in rocket_to_hit.interception_points:
                self.strategy = p
                if ((p[0] - t)*6 > np.abs(p[1] - ang)): # p[0] absolute shooting time. p[1] angle to shoot
                    break
                self.strategy = None
            if self.strategy is None:
                self.r_city_list.remove(rocket_to_hit)

        if self.strategy is None:
            if (ang > 45):
                return RIGHT
            else:
                return LEFT
        # here self.strategy is not None
        if self.strategy[1] > ang:
            action = RIGHT
        elif self.strategy[1] < ang:
            action = LEFT
        elif self.strategy[0] == t:
            action = SHOOT
            self.strategy = None
            self.r_city_list.remove(self.r_city_list[0])
        else:
            action = STRAIGHT

        # remove rockets that landed
        for r in self.r_city_list:
            if r.path[-1][2] <= t:
                self.r_city_list.remove(r)
        for r in self.r_field_list:
            if r.path[-1][2] <= t:
                self.r_field_list.remove(r)
        return action


# class MyInterceptor2():
#     def __init__(self):
#         self.r_paths_city = []
#         self.r_paths_field = []
#         self.t_and_ang_city_all = []
#         self.t_and_ang_field_all = []
#         self.t_and_ang_city_all_copy = []
#         self.t_and_ang_field_all_copy = []
#         self.strategy = None
#         self.intr_path_dict = {ang: np.array(get_interceptor_full_path(ang, 0)[:-1]) for ang in range(-84, 90, 6) }
#
#     def is_new_rocket(self, r_locs):
#         if len (r_locs) == 0:
#             return False
#         x0,y0 = 4800,0
#         x1,y1 = r_locs[-1]
#         #I calculated that in second step the distance is more than 259,
#         # while in first step it is less than 346. hence we set the threshold to 300.
#         # of course it is not accurate!!!
#         dist = np.sqrt((x1-x0)**2 + (y1-y0)**2)
#         if dist < 300:
#             return True
#         return False
# #Q: Why do we need to create a new list? Why not using the same list and just updating the last element
#     # (actually, it can be much faster if we treat the list as 2d nparray)
#     def update_times_in_r_path(self, r_paths):## Rafi
#         new_r_paths = []
#         for r_path in r_paths:
#             new_r_path = []
#             for (x, y, t) in r_path:
#                 new_r_path.append((x, y, t-1))
#             new_r_paths.append(new_r_path)
#         return new_r_paths
#
#     def update_intr_lists(self, intr_lists_all):
#         return
#
#     def calculate_action(self, r_locs, i_locs, c_locs, ang, score, t):
#
#         if self.is_new_rocket(r_locs):
#             x1, y1 = r_locs[-1]
#             new_rocket_path, city_hit = get_rocket_full_path_and_city_hit(4800, 0, x1, y1, c_locs, t)
#             if city_hit:
#                 self.r_paths_city.append(new_rocket_path)
#             else:
#                 self.r_paths_field.append(new_rocket_path)
#         #Q:  this part is the most time comsuming. If we can update instead calclulating then we can save lots of time.
#         for pth in self.r_paths_city:
#             t_and_ang_city = get_interception_times_and_angles(ang, pth, self.intr_path_dict)
#             self.t_and_ang_city_all.append(t_and_ang_city)
#         for pth in self.r_paths_field:
#             t_and_ang_field = get_interception_times_and_angles(ang, pth, self.intr_path_dict)
#             self.t_and_ang_field_all.append(t_and_ang_field)
#         #Eran - Started preparing infrastructir for updating the list to save time. need to understand how to
#         #Something like adding or deleting a coordinate  + updating times.
#         self.update_intr_lists (self.t_and_ang_field_all_copy)
#         self.update_intr_lists (self.t_and_ang_city_all_copy)
#
#         cur_game_map = game_map.build_game_map(self.t_and_ang_city_all, self.t_and_ang_field_all, ang) ##Rafi
#
#         if self.strategy is None and len (self.t_and_ang_city_all) > 0:
#             t_and_ang_city = self.t_and_ang_city_all[0]
#             self.strategy = t_and_ang_city[0]
#
#         if self.strategy is None:
#             if (ang > 45):
#                 return RIGHT
#             else:
#                 return LEFT
#         #here self.strategy is not None
#         if self.strategy[1] > ang:
#             action = RIGHT
#             self.strategy[0] -= 1
#         elif self.strategy[1] < ang:
#             action = LEFT
#             self.strategy[0] -= 1
#         elif self.strategy[0] == 0:
#             action = SHOOT
#             self.strategy = None
#         else:
#             action = STRAIGHT
#             self.strategy[0] -= 1
#         #update paths for t+1
#         for p in self.r_paths_city:
#             p.remove(p[0])
#         # while ([] in self.r_paths_city): #clean empty paths
#         #     self.r_paths_city.remove([])
#
#         self.r_paths_city = self.update_times_in_r_path(self.r_paths_city)## Rafi
#         self.r_paths_field = self.update_times_in_r_path(self.r_paths_field)## Rafi
#         self.t_and_ang_city_all_copy = self.t_and_ang_city_all_copy
#         self.t_and_ang_field_all_copy = self.t_and_ang_field_all
#         self.t_and_ang_city_all = []## Rafi
#         self.t_and_ang_field_all = []## Rafi
#         return action


def my_main():
    Init()
    my_intr = MyInterceptor()
    r_locs = i_locs = c_locs = []
    ang = score = 0
    for stp in range(1000):
        if stp % 1 == 0:
            print ("step", stp, "score", score, "rockets", len(r_locs))
        action_button = my_intr.calculate_action(r_locs, i_locs, c_locs, ang, score, stp)
        #if action_button == SHOOT:
        #    action_button = STRAIGHT ## Rafi
        r_locs, i_locs, c_locs, ang, score = Game_step(action_button)
       # Draw()
    print (score)

import cProfile

if __name__ == '__main__':
    #cProfile.run('my_main()')
    my_main()


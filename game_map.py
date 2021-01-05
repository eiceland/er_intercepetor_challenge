import numpy as np


GREEN = (0, 255, 0)
CITY_COLORS = np.asarray([(0, 0, 255), (255, 0, 0), (0, 128, 128), (128, 0, 128), (128, 128, 0), (0, 0, 64), (0, 64, 0), (64, 0, 0), (0, 64, 64), (64, 0, 64), (64, 64, 0), (0, 0, 32), (0, 32, 0), (32, 0, 0), (0, 32, 32), (32, 0, 32), (32, 32, 0)]).astype("uint8")
FIELD_COLORS = np.ceil(np.asarray(CITY_COLORS)/16).astype("uint8")


def ang2coord(ang):
    res = (np.floor(ang + 90) / 6).astype(int)
    # print("ang: " + str(ang) + " to " + str(res))
    return res


class GameMap:
    def __init__(self, shoot_interval):
        self.max_time = 400
        self.max_range = 10000
        self.angs_options = 31
        self.game_map = np.zeros((self.max_time+1, self.angs_options, 3))
        self.game_dict = {(t, a): {} for t in range(self.max_time) for a in range(self.angs_options)}
        self.color_dict = {'empty':[0, 0, 0]}
        self.time = 0
        self.shoot_interval = shoot_interval

    def get_rocket_by_id(self, rockets_list, id):
        for r in rockets_list:
            if r.id == id:
                return r

    def get_color(self, id, city_hit):
        if city_hit:
            COLOR = CITY_COLORS[id % len(CITY_COLORS)]
        else:
            COLOR = FIELD_COLORS[id % len(FIELD_COLORS)]
        return COLOR

    def update_map(self, cur_time, rockets_list, new_rockets, cur_ang, time_since_shoot):
        self.time = cur_time
        self.game_map[:-1] = self.game_map[1:]
        self.game_map[-1] = self.color_dict['empty']

        for a in range(self.angs_options):
            self.game_dict[((cur_time-1) % self.max_time, a)] = {}

        for id in new_rockets:
            r = self.get_rocket_by_id(rockets_list, id)
            self.color_dict[id] = self.get_color(id, r.city_hit)
            for p in r.interception_points:
                if (p[0] < cur_time):
                    continue
                if (p[0] >= cur_time + self.max_time):
                    continue
                dict_key = (p[0] % self.max_time, ang2coord(p[1]))
                self.game_dict[dict_key][id] = p[2]
                cur_dict = self.game_dict[dict_key]
                min_id = min(cur_dict, key=lambda k: cur_dict[k])
                ## game map: line 0 - the actor, line 1 - rocket to hit if I shoot now.
                self.game_map[p[0] - cur_time + 1, ang2coord(p[1])] = self.color_dict[min_id]

        ## zero non-shooting rows:
        if time_since_shoot < self.shoot_interval:
            time_to_shoot_option = self.shoot_interval - time_since_shoot
            self.game_map[:time_to_shoot_option+1, :, :] = 0

        ## add player:
        self.game_map[0, :, :] = 0
        self.game_map[0, ang2coord(cur_ang), :] = GREEN
        return self.game_map

    def delete_rockets_path_after_shoot(self, rockets_list):
        agent_loc = np.nonzero(self.game_map[0])[0][0]
        dict_key = (self.time % self.max_time, agent_loc)
        cur_dict = self.game_dict[dict_key]
        if (len(cur_dict) == 0):
            return []
        min_id = min(cur_dict, key=lambda k: cur_dict[k])
        rockets_to_remove = []
        for id in cur_dict:
            if cur_dict[id] == cur_dict[min_id]:
                rockets_to_remove.append(id)
        for id in rockets_to_remove:
            r = self.get_rocket_by_id(rockets_list, id)
            for p in r.interception_points:
                if (p[0] < self.time):
                    continue
                if (p[0] >= self.time + self.max_time):
                    continue
                dict_key = (p[0] % self.max_time, ang2coord(p[1]))
                del self.game_dict[dict_key][id]
                cur_dict = self.game_dict[dict_key]
                if len(cur_dict) == 0:
                    self.game_map[p[0] - self.time + 1, ang2coord(p[1])] = self.color_dict['empty']
                else:
                    min_id = min(cur_dict, key=lambda k: cur_dict[k])
                    self.game_map[p[0] - self.time + 1, ang2coord(p[1])] = self.color_dict[min_id]

        return rockets_to_remove


def main():
    a = 1


if __name__ == '__main__':
    main()

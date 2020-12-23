import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


GREEN = (0, 255, 0)
CITY_COLORS = np.asarray([(0, 0, 255), (255, 0, 0), (0, 128, 128), (128, 0, 128), (128, 128, 0), (0, 0, 64), (0, 64, 0), (64, 0, 0), (0, 64, 64), (64, 0, 64), (64, 64, 0), (0, 0, 32), (0, 32, 0), (32, 0, 0), (0, 32, 32), (32, 0, 32), (32, 32, 0)]).astype("uint8")
FIELD_COLORS = np.ceil(np.asarray(CITY_COLORS)/8).astype("uint8")


def ang2coord(ang):
    return (np.floor(ang + 90) / 6).astype(int)


def build_game_map(rockets_list, cur_ang, cur_time):
    disp = True # False
    max_time = 400
    max_range = 10000
    angs_options = 31
    game_map = np.zeros((max_time, angs_options, 3))
    inter_ranges_map = np.zeros((max_time, angs_options)) + max_range
    inter_ranges_map[0, ang2coord(cur_ang)] = 0
    temp_game_map = np.zeros((max_time, angs_options))
    temp_inter_ranges_map = np.zeros((max_time, angs_options))

    for rocket in rockets_list:
        id = rocket.id
        if rocket.city_hit:
            COLOR = CITY_COLORS[id % len(CITY_COLORS)]
        else:
            COLOR = FIELD_COLORS[id % len(FIELD_COLORS)]

        # t_and_ang = rocket.interception_points
        # for t, ang, inter_range in t_and_ang:
        #     if t+1 - cur_time < 1:
        #         continue
        #     ang_coord = ang2coord(ang)
        #     if inter_ranges_map[t+1 - cur_time, ang_coord] > inter_range:
        #         game_map[t+1 - cur_time, ang_coord, :] = COLOR

        ## fast version:
        if len(rocket.interception_points) == 0:
            print("warning: rocket without interceptions. ")
            print("rocket id: " + str(rocket.id))
            print("rocket track: " + str(rocket.path))
            continue
        t_and_ang = np.asarray(rocket.interception_points)
        temp_game_map[:] = 0
        temp_inter_ranges_map[:] = max_range

        t_and_ang[:, 0] -= cur_time
        t_and_ang = t_and_ang[t_and_ang[:, 0] > 0]
        t_and_ang = t_and_ang[t_and_ang[:, 0] < max_time]

        angs = ang2coord(t_and_ang[:, 1])
        temp_game_map[t_and_ang[:, 0], angs] = 1
        temp_inter_ranges_map[t_and_ang[:, 0], angs] = t_and_ang[:, 2]

        closest_inter = np.argmin(np.dstack((temp_inter_ranges_map, inter_ranges_map)), 2)
        for i in range(3):
            game_map_i = game_map[:, :, i]
            chosen = [closest_inter == 0]
            game_map_i[chosen] = temp_game_map[chosen] * COLOR[i]
            game_map[:, :, i] = game_map_i
        a=1

    game_map[0, :, :] = 0
    game_map[0, ang2coord(cur_ang), :] = GREEN

    if disp:
        im = game_map.astype("uint8")
        im = cv.resize(im, (angs_options*20, max_time*3), interpolation=cv.INTER_NEAREST)
        im = 255 - im
        im = im.astype("uint8")
        im = np.vstack((cv.resize(im[:1, :], (angs_options*20, 10)), im))
        plt.imshow(im)
        plt.title("time: " + str(cur_time) + " rockets: " + str(len(rockets_list)))
        plt.pause(0.0001)
    return game_map


def main():
    a = 1


if __name__ == '__main__':
    main()

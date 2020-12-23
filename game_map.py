import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


GREEN = (0, 255, 0)
CITY_COLORS = [(0, 0, 255), (255, 0, 0), (0, 128, 128), (128, 0, 128), (128, 128, 0), (0, 0, 64), (0, 64, 0), (64, 0, 0), (0, 64, 64), (64, 0, 64), (64, 64, 0), (0, 0, 32), (0, 32, 0), (32, 0, 0), (0, 32, 32), (32, 0, 32), (32, 32, 0)]
FIELD_COLORS = np.asarray(CITY_COLORS)/8


def ang2coord(ang):
    return int(np.floor(ang + 90) / 6)


def build_game_map(rockets_list, cur_ang, cur_time):
    disp = False
    game_map = np.zeros((400, 31, 3))
    inter_ranges_map = np.zeros((400, 31)) + 10000
    game_map[0, ang2coord(cur_ang), :] = GREEN
    inter_ranges_map[0, ang2coord(cur_ang)] = 0

    for rocket in rockets_list:
        id = rocket.id
        if rocket.city_hit:
            COLOR = CITY_COLORS[id % len(CITY_COLORS)]
        else:
            COLOR = FIELD_COLORS[id % len(FIELD_COLORS)]

        t_and_ang = rocket.interception_points
        for t, ang, inter_range in t_and_ang:
            if t+1 - cur_time < 1:
                continue
            ang_coord = ang2coord(ang)
            if inter_ranges_map[t+1 - cur_time, ang_coord] > inter_range:
                game_map[t+1 - cur_time, ang_coord, :] = COLOR
    if disp:
        im = game_map.astype("uint8")
        im = cv.resize(im, (620, 1200), interpolation=cv.INTER_NEAREST)
        im = 255 - im
        im = im.astype("uint8")
        im = np.vstack((cv.resize(im[:1, :], (620, 10)), im))
        plt.imshow(im)
        plt.title("time: " + str(cur_time) + " rockets: " + str(len(rockets_list)))
        plt.pause(0.0001)
    return game_map


def main():
    a = 1


if __name__ == '__main__':
    main()

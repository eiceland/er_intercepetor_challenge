import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os

GREEN = (0, 255, 0)
COLORS = [(0, 0, 255), (255, 0, 0), (0, 128, 128), (128, 0, 128), (128, 128, 0), ]
GRAY = [128, 128, 128]


def ang2coord(ang):
    return int(np.floor(ang + 90) / 6)


def build_game_map(t_and_ang_city_all, t_and_ang_field_all, cur_ang):

    game_map = np.zeros((400, 31, 3))
    inter_ranges_map = np.zeros((400, 31)) + 10000
    game_map[0, ang2coord(cur_ang), :] = GREEN
    for i, t_and_ang_city in enumerate(t_and_ang_city_all):
        COLOR = COLORS[i%len(COLORS)]
        for t, ang, inter_range in t_and_ang_city:
            ang_coord = ang2coord(ang)
            if inter_ranges_map[t+1, ang_coord] > inter_range:
                game_map[t+1, ang_coord, :] = COLOR
    for t_and_ang_field in t_and_ang_field_all:
        for t, ang, inter_range in t_and_ang_field:
            ang_coord = ang2coord(ang)
            if inter_ranges_map[t+1, ang_coord] > inter_range:
                game_map[t+1, ang2coord(ang), :] = GRAY

    plt.imshow(game_map.astype("uint8"))
    plt.pause(0.0001)
    return game_map


def main():
    a = 1


if __name__ == '__main__':
    main()

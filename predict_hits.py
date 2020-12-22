import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os

UNKNOWN = -1
HIT = 1
NO_HIT = 0

#given two succesive points in a path, compute the next n points.
def get_path(x0, y0, x1, y1, n, fric = 5e-7, dt = 0.2 , g = 9.8):
    dx, dy = x1 - x0, y1 - y0
    vx, vy = dx / dt, dy / dt
    x, y = x1, y1
    path = []
    for _ in range(n):
        v_loss = (vx * 2 + vy * 2) * fric * dt
        vx = vx * (1 - v_loss)
        vy = vy * (1 - v_loss) - g * dt
        x = x + vx * dt
        y = y + vy * dt
        path.append((x, y))
        if y < 0:
            break
    return path


def predict_hits(tracks, city_locs):
    display = True
    ground_level = 0
    city_width = city_locs[0][1]
    hits = np.zeros(len(tracks))
    for i, track in enumerate(tracks):
        if len(track.points) < 2:
            hits[i] = UNKNOWN
            continue
        points = np.asarray(track.points)
        t0 = points[0][0] + 1
        n = 1000
        x0 = 4800
        y0 = 0
        preds_xy = get_path(x0, y0, points[0][1], points[0][2], n, fric=5e-7, dt=0.2, g=9.8)
        preds_xy = np.asarray(preds_xy)
        preds = np.zeros((len(preds_xy), 3))
        preds[:, 0] = np.arange(t0, t0 + len(preds_xy))
        preds[:, 1] = preds_xy[:, 0]
        preds[:, 2] = preds_xy[:, 1]

        if display:
            d3 = False
            if d3:
                fig = plt.figure()
                ax = plt.axes(projection='3d')
                ax.scatter3D(points[:, 0], points[:, 1], points[:, 2])
                preds = np.asarray(preds)
                ax.scatter3D(preds[:, 0], preds[:, 1], preds[:, 2])
            else:
                preds = np.asarray(preds)
                plt.subplot(1, 2, 1)
                plt.plot(points[:, 0], points[:, 1], 'og')
                plt.plot(preds[:, 0], preds[:, 1], '.r')
                plt.subplot(1, 2, 2)
                plt.plot(points[:, 0], points[:, 2], 'og')
                plt.plot(preds[:, 0], preds[:, 2], '.r')
                plt.pause(0.001)
                break

        hit = NO_HIT
        x = preds[-2][1]
        for city in city_locs:
            if np.abs(x - city[0]) < city_width:
                hit = HIT
        hits[i] = hit

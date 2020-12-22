import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


class Track:

    def __init__(self):
        self.points = []
        self.last_point = []
        self.last_last_point = []
        self.last_direc = []
        self.last_size = [1000000, 1000000]
        self.last_imnum = -1


    def add_point(self, point, im_num):
        self.points.append([im_num, point[0], point[1]])
        self.last_last_point = self.last_point
        self.last_point = np.asarray(point)
        self.last_imnum = im_num


class Tracker:

    def __init__(self):
        self.tracks = []
        self.last_image = []
        self.global_disparity = [0, 0]
        self.use_global_tracker = False

    def add_point(self, point, im_num):
        thresh = 1000
        mn_dist = 10000
        for track in self.tracks:
            if track.last_imnum == im_num:
                continue
            dist = np.linalg.norm(np.asarray(track.last_point) - np.asarray(point))
            if dist < mn_dist:
                mn_dist = dist
                mn_dist_track = track
        if mn_dist < thresh:
            mn_dist_track.add_point(point, im_num)
        else:
            new_track = Track()
            new_track.add_point(point, im_num)
            self.tracks.append(new_track)

    def display(self):
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        colors = ['Reds', 'Greens', 'Blues'] #'rgbcmyk'
        for i, track in enumerate(self.tracks):
            color = colors[i%3]
            points = np.asarray(track.points)
            ax.scatter3D(points[:, 0], points[:, 1], points[:, 2], cmap=color)
        plt.show()


    def track_image(self, detections, im_num):

        for detection in detections:
            # type = detection[0]
            roi = np.asarray(detection)
            cen = [roi[0], roi[1]]

            self.add_point(cen, im_num)




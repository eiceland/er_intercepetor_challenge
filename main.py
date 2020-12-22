from Interceptor_V2 import Init, Draw, Game_step
from tracker import Tracker
import predict_hits

if __name__ == '__main__':

    Init()
    tracker = Tracker()

    for stp in range(1000):
        if stp < 7:
            action_button = 2
        else:
            action_button = 3
        r_locs, i_locs, c_locs, ang, score = Game_step(action_button)
        tracker.track_image(r_locs, stp)
        # Draw()
        # tracker.display()
        predict_hits.predict_hits(tracker.tracks, c_locs)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

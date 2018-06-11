"""
@author: Mahmoud I.Zidan
Mod by XC
"""
import numpy as np
import os.path
import time
from sort import Sort


class Tracker:

    def __init__(self, max_age=30,min_hits=5, use_dlib = False):
        self.tracker = Sort(max_age, min_hits, use_dlib)

        if use_dlib:
            print "TRACKER: Dlib Correlation tracker activated!"
        else:
            print "TRACKER: Kalman SORT tracker activated!"


    def update(self, dets, img=None):
        # format of tracks:
        # [
        #   [x0, y0, x1, y1, tk_id], ...
        # ]
        return self.tracker.update(np.array(dets), img)


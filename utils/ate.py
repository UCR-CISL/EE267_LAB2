'''
UC Riverside
EE260: Introduction to Self-Driving Stack
Inspired by: https://github.com/aau-cns/cnspy_trajectory_evaluation
'''

import os
import sys

import numpy as np
import cv2
import time
from queue import Queue
from queue import Empty

class AbsoluteTrajectoryError:
    traj_err = None  # TrajectoryError
    traj_est = None  # Trajectory/TrajectoryEstimated
    traj_gt = None   # Trajectory

    def __init__(self, traj_est, traj_gt):
        self.traj_est = traj_est
        self.traj_gt = traj_gt
        self.traj_err = AbsoluteTrajectoryError.compute_trajectory_error(traj_est=traj_est, traj_gt=traj_gt,
                                                                         traj_err_type=traj_err_type)
        pass
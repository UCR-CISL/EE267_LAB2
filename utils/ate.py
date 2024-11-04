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

from scipy.spatial.transform import Rotation
import pdb

class AbsoluteTrajectoryError:
    traj_err = None  # TrajectoryError
    traj_est = None  # Trajectory/TrajectoryEstimated
    traj_gt = None   # Trajectory

    def __init__(self, traj_gt, traj_est):
        self.traj_est = traj_est
        self.traj_gt = traj_gt
    
    def compute_trajectory_error(self):
        """
        Compute ATE between estimated and ground truth trajectories
        Input format for each trajectory: [x, y, z, qw, qx, qy, qz]
        """

        # Ensure trajectories have same length
        assert len(self.traj_est) == len(self.traj_gt), "Trajectories must have same length"
        
        errors = []
        for est, gt in zip(self.traj_est, self.traj_gt):

            # Extract positions
            p_est = est[:3].reshape(3,1)  # [x, y, z]
            p_gt = gt[:3].reshape(3,1)    # [x, y, z]
            
            # Extract quaternions and convert to rotation matrices
            q_est = est[3:].reshape(-1)  # [qw, qx, qy, qz]
            q_gt = gt[3:].reshape(-1)    # [qw, qx, qy, qz]
            
            R_est = Rotation.from_quat([q_est[1], q_est[2], q_est[3], q_est[0]]).as_matrix()
            R_gt = Rotation.from_quat([q_gt[1], q_gt[2], q_gt[3], q_gt[0]]).as_matrix()
            
            # Compute relative transformation
            R_rel = R_gt.T @ R_est
            t_rel = R_gt.T @ (p_est - p_gt)
            
            # Compute error
            pos_error = np.linalg.norm(t_rel)
            rot_error = np.arccos((np.trace(R_rel) - 1) / 2)
            
            errors.append({
                'position_error': pos_error,
                'rotation_error': rot_error
            })
            
        return errors
    
    def get_statistics(self):
        """
        Compute statistics of the trajectory errors
        """
        pos_errors = [err['position_error'] for err in self.traj_err]
        rot_errors = [err['rotation_error'] for err in self.traj_err]
        
        stats = {
            'rmse_position': np.sqrt(np.mean(np.array(pos_errors)**2)),
            'mean_position': np.mean(pos_errors),
            'median_position': np.median(pos_errors),
            'std_position': np.std(pos_errors),
            'min_position': np.min(pos_errors),
            'max_position': np.max(pos_errors),
            
            'rmse_rotation': np.sqrt(np.mean(np.array(rot_errors)**2)),
            'mean_rotation': np.mean(rot_errors),
            'median_rotation': np.median(rot_errors),
            'std_rotation': np.std(rot_errors),
            'min_rotation': np.min(rot_errors),
            'max_rotation': np.max(rot_errors)
        }
        
        return stats
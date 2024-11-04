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

if __name__ == '__main__':
    # Create dummy data
    num_poses = 100
    
    # Create ground truth trajectory
    gt_trajectories = []
    for i in range(num_poses):
        # Create a circular path for ground truth
        t = i * 2 * np.pi / num_poses
        x = np.cos(t)
        y = np.sin(t)
        z = 0.1 * t
        
        # Create a rotation (rotating around z-axis)
        rotation = Rotation.from_euler('z', t)
        quat = rotation.as_quat()  # Returns [x, y, z, w]
        # Reorder to [w, x, y, z]
        qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]
        
        pose = np.array([x, y, z, qw, qx, qy, qz]).reshape((7,1))
        gt_trajectories.append(pose)
    
    # Create estimated trajectory with some noise
    est_trajectories = []
    noise_pos = 0.1  # 10cm position noise
    noise_angle = 0.05  # Small rotation noise (radians)
    
    for gt_pose in gt_trajectories:
        # Add noise to position
        noisy_pos = gt_pose[:3] + np.random.normal(0, noise_pos, (3,1))
        
        # Add noise to rotation
        quat = gt_pose[3:].reshape(-1)  # Reshape to 1D array
        
        # Create noise rotation using euler angles
        noise_euler = np.random.normal(0, noise_angle, 3)  # Random rotation around each axis
        noise_rot = Rotation.from_euler('xyz', noise_euler)
        
        original_rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])  # [x,y,z,w] format
        noisy_rot = noise_rot * original_rot
        noisy_quat = noisy_rot.as_quat()  # [x,y,z,w] format
        
        # Reorder quaternion to [w, x, y, z]
        noisy_pose = np.vstack((noisy_pos, 
                               np.array([[noisy_quat[3]], 
                                       [noisy_quat[0]], 
                                       [noisy_quat[1]], 
                                       [noisy_quat[2]]]))
                             )
        est_trajectories.append(noisy_pose)
    
    # Compute ATE
    ate = AbsoluteTrajectoryError(gt_trajectories, est_trajectories)
    ate.traj_err = ate.compute_trajectory_error()
    stats = ate.get_statistics()
    
    # Print results
    print("\nTrajectory Error Statistics:")
    print(f"RMSE Position Error: {stats['rmse_position']:.3f} meters")
    print(f"Mean Position Error: {stats['mean_position']:.3f} meters")
    print(f"Median Position Error: {stats['median_position']:.3f} meters")
    print(f"Std Position Error: {stats['std_position']:.3f} meters")
    print(f"\nRMSE Rotation Error: {stats['rmse_rotation']:.3f} radians")
    print(f"Mean Rotation Error: {stats['mean_rotation']:.3f} radians")
    print(f"Median Rotation Error: {stats['median_rotation']:.3f} radians")
    print(f"Std Rotation Error: {stats['std_rotation']:.3f} radians")

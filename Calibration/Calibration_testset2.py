#!/usr/bin/env python
import sys

import numpy as np

sys.path.append('./')
import random

import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils.transformations import quaternion_matrix

"""
Param
---------
rvecs: Roatation matrix from maker to camera frame
tvecs: Translation component of the transformation from marker frame to camera frame
pvecs
"""
translations_all = [[0.048, 0.055, 0.111],[0.058, 0.077, 0.123],
                    [0.037, 0.024, 0.107],[0.022, 0.064, 0.135],
                    [0.039, 0.071, 0.166],[0.050, 0.076, 0.172],
                    [0.054, 0.081, 0.179], [0.075, 0.083, 0.199],
                    [0.071, 0.078, 0.212], [0.108, -0.002, 0.185],
                    [0.030, 0.049, 0.143],  [0.089, -0.002, 0.180],
                    [0.126, -0.017, 0.190], [0.071, 0.032, 0.224],
                    [0.034, 0.111, 0.141], [0.047, 0.009, 0.119],
                    [0.083, -0.042, 0.144],[0.033, 0.056, 0.116],
                     [0.013, 0.092, 0.191],  [0.035, 0.088, 0.219],
                     [0.006, -0.011, 0.195]
				]
quats_all = [  [0.638, 0.765, -0.027, -0.078], [0.671, 0.736, -0.066, -0.059],
                 [0.429, 0.898, 0.004, -0.103], [0.514, 0.856, -0.042, -0.038],
                 [0.518, 0.850, -0.079, -0.053],[0.510, 0.852, -0.101, -0.064],
                 [0.515, 0.847, -0.117, -0.064], [0.487, 0.856, -0.150, -0.093],
                 [0.509, 0.845, -0.138, -0.088], [0.593, 0.776, -0.045, -0.212],
                  [0.353, 0.932, -0.057, -0.065],[0.288, 0.939, -0.030, -0.185],
                  [0.255, 0.933, -0.028, -0.250], [0.394, 0.906, -0.085, -0.131],
                  [0.643, 0.759, -0.101, 0.011],[0.753, 0.643, 0.047, -0.132],
                  [0.631, 0.739, 0.050, -0.231],[0.427, 0.900, -0.050, -0.065],
                   [0.603, 0.794, -0.074, 0.021],  [0.622, 0.777, -0.095, -0.008],
                   [0.570, 0.816, 0.067, -0.073]
	 ]
p_fix_all = [ [-0.055, 0.031, 0.290] ]*21
assert len(p_fix_all) == len(quats_all) and len(quats_all) == len(translations_all)
consensus_set = []
RANSAC_times = 500
for time in range(RANSAC_times):
    sample_num = 18#len(p_fix)
    idx_set = [i for i in range(len(p_fix_all))]
    selected_idx = random.sample(idx_set, sample_num) if time < RANSAC_times - 1 else rank[:sample_num]
    translations = np.array(translations_all)[selected_idx]
    quats = np.array(quats_all)[selected_idx]
    p_laser = np.array(p_fix_all)[selected_idx]
    local_points = []
    for i in range(sample_num):
        rotmatrix = quaternion_matrix(quats[i])[:3,:3]
        local_point = np.matmul(rotmatrix ,np.array(p_laser[i])[:,None]) \
            + np.array(translations[i])[:,None]
        local_points.append(local_point[:,0])
    local_points = np.array(local_points)
    local_points_shifted = local_points - np.mean(local_points, axis = 0, keepdims = True)
    u,s,vh = np.linalg.svd(local_points_shifted)
    direc_vec = vh[0]
    data_mean = np.mean(local_points, axis = 0)
    if time < RANSAC_times - 2:
        errs = []
        for i in range(len(local_points)):
            coefficient,err,_,_ = np.linalg.lstsq(direc_vec[:,None], local_points[i][:,None] - data_mean[:,None],rcond=None)
            best_fit = direc_vec * coefficient[0][0] + data_mean
            err = np.linalg.norm(local_points[i] - best_fit)
            if err < 0.002:
                consensus_set.append(selected_idx[i])
            errs.append(err)
        print('{} residual error is: '.format(time), np.mean(errs), np.max(errs))
    if time == RANSAC_times - 2:
        votes = []
        consensus_set=np.array(consensus_set,dtype=np.int32)
        for real_idx in range(len(p_fix_all)):
            votes.append(np.sum(consensus_set == real_idx))
        rank = np.argsort(votes)[::-1]
        print('votes: ', votes)
        print('selected sample pairs idx: ', rank[:sample_num])
    if time == RANSAC_times - 1:
    # Visualization
        colors = ['b','g','r','c','m','y','k','brown','gold','teal','plum']
        fig1 = plt.figure(1)
        ax = fig1.gca(projection='3d')
        errs = []
        coefficients = []
        for i in range(len(local_points)):
            coefficient,err,_,_ = np.linalg.lstsq(direc_vec[:,None], local_points[i][:,None] - data_mean[:,None],rcond=None)
            best_fit = direc_vec * coefficient[0][0] + data_mean
            coefficients.append(coefficient[0][0])
            err = np.linalg.norm(local_points[i] - best_fit)
            errs.append(err)
            ax.scatter(local_points[i][0],local_points[i][1],local_points[i][2], marker='o',color = 'b')
            ax.scatter(best_fit[0],best_fit[1],best_fit[2], marker='^', color = 'r')
            if i > 0:
                ax.plot(np.array([best_fit[0],lst_best_fit[0]]),np.array([best_fit[1],lst_best_fit[1]]),np.array([best_fit[2],lst_best_fit[2]]),'red')
            lst_best_fit = best_fit
        ax.scatter(0.17,0.17,-0.2, marker='^', color = 'r')
        print('residual error is: ', np.mean(errs), np.max(errs))
        import random

        import scipy.stats as st
        sampled_errs = []
        for i in range(100):
            _errs = random.sample(errs,int(0.7*len(errs)))
            sampled_errs.append(np.mean(_errs))
        sampled_errs = np.asarray(sampled_errs)
        lower, upper = st.t.interval(alpha=0.95, df=len(sampled_errs)-1,
                                 loc=np.mean(sampled_errs),
                                 scale=st.sem(sampled_errs))
        print('error confidence interval: ', lower, upper)
        print('fitted results: ', data_mean, direc_vec )
        # Estimate start point
        coefficient,err,_,_ = np.linalg.lstsq(direc_vec[:,None], -1*data_mean[:,None],rcond=None)
        start_point = direc_vec[:,None] * coefficient[0][0] + data_mean[:,None]
        print("estimated start point: ", start_point[:,0])
        ax.set_xlabel('X Label (m)')
        ax.set_ylabel('Y Label (m)')
        ax.set_zlabel('Z Label (m)')
        plt.show()
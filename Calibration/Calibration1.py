#!/usr/bin/env python
import sys

import numpy as np

sys.path.append('./')
import random

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
translations_all = [[0.050, 0.074, 0.252], [0.058, 0.094, 0.286],
				[0.020, 0.122, 0.308], [0.021, 0.102, 0.231],
				[0.012, 0.061, 0.247], [-0.042, 0.052, 0.202],
				[-0.014, 0.044, 0.318], [0.032, 0.168, 0.188],
				[0.072, 0.191, 0.212], [-0.033, 0.016, 0.287],
				[0.014, 0.148, 0.262], [-0.123, -0.050, 0.167]]
TRANS = [[0.058, 0.079, 0.269], [-0.034, 0.013, 0.275]]
quats_all = [[0.305, 0.952, 0.003, 0.001], [0.025, 0.997, -0.062, -0.039],
		 [-0.047, 0.993, -0.109, -0.010], [0.147, 0.987, -0.047, 0.039],
		 [-0.043, 0.998, -0.021, 0.031], [-0.118, 0.986, -0.035, 0.110],
		 [-0.230, 0.972, -0.041, 0.036], [-0.510, 0.830, -0.209, -0.084],
		 [-0.517, 0.817, -0.205, -0.152], [-0.451, 0.887, -0.068, 0.078],
		 [-0.255, 0.952, -0.169, -0.025], [-0.396, 0.884, -0.027, 0.247]]
QUATS = [[-0.492, 0.864, -0.079, -0.063], [-0.470, 0.875, -0.075, 0.090]]
p_fix_all = [[-0.055, -0.011, 0.389]] * 7 + [[-0.054, -0.011, 0.390]] * 5
assert len(p_fix_all) == len(quats_all) and len(quats_all) == len(translations_all)
consensus_set = []
RANSAC_times = 501
for time in range(RANSAC_times):
    sample_num = 9#len(p_fix)
    idx_set = [i for i in range(len(p_fix_all))] 
    selected_idx = random.sample(idx_set, sample_num) if time < RANSAC_times - 1 else rank[:sample_num]
    translations = np.array(translations_all)[selected_idx]
    quats = np.array(quats_all)[selected_idx]
    p_laser = np.array(p_fix_all)[selected_idx]
    local_points = []
    for i in range(sample_num):
        rotmatrix = quaternion_matrix(quats[i])[:3,:3]
        local_point = np.matmul(rotmatrix ,np.array(p_laser[i])[:,None]) + np.array(translations[i])[:,None]
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

# fitted results:  [ 0.08478048  0.03536605 -0.1202362 ] [-0.19600426 -0.05415381  0.97910658]
# estimated start point:  [0.05876608 0.02799237 0.01331244]
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

translations_all = [ [0.013, 0.031, 0.101],[0.006, 0.031, 0.072],
                    [-0.002, 0.043, 0.120],[0.009, 0.009, 0.139],
                    [-0.022, -0.017, 0.157], [0.005, 0.001, 0.172],
                     [-0.026, -0.026, 0.179],[-0.029, -0.042, 0.193],
                     [0.035, -0.012, 0.198],[0.019, 0.044, 0.175],
                     [0.008, 0.021, 0.147], [0.005, 0.047, 0.121],
                     [0.003, 0.054, 0.091],[0.005, 0.025, 0.070],
                     [-0.018, 0.016, 0.088], [-0.033, 0.067, 0.144],
                      [-0.049, -0.031, 0.121]


				]


quats_all = [  [-0.373, 0.925, -0.071, -0.018], [-0.422, 0.903, -0.081, 0.018],
                [-0.473, 0.874, -0.105, 0.020], [-0.463, 0.886, -0.032, 0.024],
                [-0.447, 0.889, -0.012, 0.099],[-0.459, 0.888, -0.014, 0.035],
                [-0.399, 0.911, 0.013, 0.101],[-0.379, 0.918, 0.048, 0.112],
                [-0.676, 0.734, 0.021, 0.053],[-0.688, 0.722, -0.073, 0.007],
                [-0.617, 0.782, -0.072, 0.045],[-0.199, 0.976, -0.090, -0.025],
                [-0.144, 0.984, -0.104, -0.021], [-0.474, 0.877, -0.075, 0.030],
                [-0.424, 0.900, -0.072, 0.071],[-0.410, 0.896, -0.166, 0.046],
                 [-0.461, 0.871, -0.020, 0.167]


	 ]


p_fix_all = [[-0.036, 0.039, 0.264]] *17

assert len(p_fix_all) == len(quats_all) and len(quats_all) == len(translations_all)

consensus_set = []
RANSAC_times = 500
for time in range(RANSAC_times):
    sample_num = 12 #len(p_fix_all)
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
        ax.set_yticks([-0.1 + i*0.025 for i in range(15) if -0.1 + i*0.025 < 0.04])
        ax.set_xticks([-0.06 + i*0.025 for i in range(15) if -0.1 + i*0.025 < 0.04])
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
        ax.scatter(-0.08,-0.06,-0.2, marker='^', color = 'r')

                
        
        print('residual error is: ', np.mean(errs), np.std(errs))
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
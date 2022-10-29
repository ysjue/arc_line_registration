#!/usr/bin/env python
import numpy as np
import sys
sys.path.append('./')
from utils.transformations import quaternion_matrix
import cv2   
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
"""
Param
---------
rvecs: Roatation matrix from maker to camera frame
tvecs: Translation component of the transformation from marker frame to camera frame
pvecs
"""

translations_all = [[0.072, 0.050, 0.126], [0.103, 0.102, 0.174],
				[0.087, 0.092, 0.175],[0.047, 0.171, 0.202],
				[0.052, 0.125, 0.249],[0.087, 0.090, 0.164],
				[0.079, 0.065, 0.119], [0.088, 0.038, 0.115],
				[0.113, 0.053, 0.107],[0.163, 0.112, 0.125],
				[0.051, -0.014, 0.098],[0.072, 0.095, 0.158],
				 [0.063, 0.086, 0.153],[0.096, 0.012, 0.121],
				 [0.074, 0.196, 0.183],[0.140, 0.115, 0.195]
				]

quats_all = [[0.534, 0.792, -0.084, -0.286], [0.468, 0.823, -0.168, -0.273],
		[0.048, 0.923, -0.212, -0.318], [0.248, 0.926, -0.262, -0.109],
		 [0.196, 0.953, -0.189, -0.130],[0.281, 0.918, -0.161, -0.228],
		 [0.464, 0.850, -0.118, -0.219], [0.827, 0.540, -0.056, -0.145],
		 [0.407, 0.857, -0.133, -0.288],[0.383, 0.835, -0.244, -0.310],
		 [0.662, 0.716, 0.008, -0.219],[0.060, 0.952, -0.179, -0.242],
		 [0.594, 0.787, -0.099, -0.136], [0.411, 0.864, -0.057, -0.286],
		 [0.489, 0.828, -0.270, -0.051], [0.575, 0.773, -0.192, -0.190]
	 ]

p_fix_all = [[-0.129, 0.048, 0.286]] * 3 \
		+ [ [-0.089, 0.023, 0.293]]*13

assert len(p_fix_all) == len(quats_all) and len(quats_all) == len(translations_all)

consensus_set = []
RANSAC_times = 501
for time in range(RANSAC_times):
    sample_num = 12#len(p_fix)
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
            if err < 0.003:
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
        import scipy.stats as st
        import random
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
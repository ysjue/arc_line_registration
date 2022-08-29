#!/usr/bin/env python
import numpy as np
import sys
sys.path.append('./')
from utils.transformations import quaternion_matrix
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
translations_all = [[-0.082, -0.000, 0.282],[-0.087, -0.006, 0.297],
	 [-0.093, -0.008, 0.299], [-0.089, -0.005, 0.312],
	 [-0.090, -0.000, 0.314], [-0.089, -0.004, 0.328],
	 [-0.098, 0.020, 0.282], [-0.083, 0.040, 0.290],
	  [-0.080, 0.042, 0.300],[-0.078, 0.043, 0.309],
	  [-0.134, -0.043, 0.248], [-0.100, -0.009, 0.262],
	  [-0.112, -0.021, 0.264],[-0.099, -0.022, 0.268],
	  [-0.130, -0.028, 0.242],[-0.069, -0.006, 0.309],
	  [-0.084, -0.031, 0.307], [-0.086, -0.032, 0.294],
	  [-0.095, -0.033, 0.273],[-0.130, -0.047, 0.227],
	   [-0.076, -0.028, 0.293],[-0.068, -0.010, 0.301],
	   [-0.070, 0.013, 0.345], [-0.072, 0.030, 0.356],
	    [-0.057, 0.004, 0.141],[-0.021, -0.018, 0.159],
	    [-0.109, -0.033, 0.247]
]
quats_all = [[0.240, 0.932, -0.102, 0.252], [0.261, 0.924, -0.113, 0.255],
	 [0.283, 0.923, -0.105, 0.239], [0.277, 0.918, -0.094, 0.268],
	 [0.279, 0.921, -0.078, 0.259], [0.281, 0.913, -0.090, 0.283],
	 [-0.004, 0.963, -0.137, 0.232], [-0.225, 0.963, 0.005, 0.149],
	 [-0.221, 0.962, 0.000, 0.160],[-0.216, 0.962, 0.001, 0.167],
	 [0.408, 0.866, -0.251, 0.145],[0.301, 0.924, -0.114, 0.208],
	 [0.446, 0.865, -0.116, 0.199],[0.476, 0.836, -0.092, 0.255],
	 [0.501, 0.835, -0.161, 0.160],[0.176, 0.918, -0.139, 0.328],
	 [0.238, 0.896, -0.225, 0.301],[0.238, 0.907, -0.218, 0.272],
	 [0.225, 0.924, -0.216, 0.221], [0.529, 0.798, -0.212, 0.196],
	 [0.363, 0.858, -0.137, 0.337],[0.290, 0.888, -0.093, 0.345],
	 [0.259, 0.882, 0.059, 0.389],[0.140, 0.914, 0.124, 0.360],
	 [0.892, 0.414, 0.068, 0.167],[0.850, 0.462, 0.093, 0.237],
	 [0.414, 0.870, -0.163, 0.212]
	 ]
p_fix_all = [[-0.144, 0.043, 0.380]] * len(quats_all)
assert len(p_fix_all) == len(quats_all) and len(quats_all) == len(translations_all)

consensus_set = []
RANSAC_times = 501
for time in range(RANSAC_times):
    sample_num = 15#len(p_fix)
    idx_set = [i for i in range(len(p_fix_all))] 
    selected_idx = random.sample(idx_set, sample_num) if time < RANSAC_times - 1 else rank[:sample_num]
    translations = np.array(translations_all)[selected_idx]
    quats = np.array(quats_all)[selected_idx]
    p_laser = np.array(p_fix_all)[selected_idx]
    local_points = []
    for i in range(sample_num):
        rotmatrix = quaternion_matrix(quats[i])[:3,:3]
        local_point = np.matmul(rotmatrix.T ,np.array(p_laser[i])[:,None]) \
            + np.matmul(rotmatrix.T , -1 *np.array(translations[i])[:,None])
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

# fitted results:  [ 0.08478048  0.03536605 -0.1202362 ] [-0.19600426 -0.05415381  0.97910658]
# estimated start point:  [0.05876608 0.02799237 0.01331244]
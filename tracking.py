import os
import numpy as np
import scipy.stats as st
import yaml
from utils.transformations import rotation_matrix, quaternion_matrix, unit_vector
from Solver.point_line_registration import homo
from scipy.optimize import least_squares

import numpy as np
import random
from utils.transformations import identity_matrix,unit_vector,rotation_matrix,quaternion_matrix, rotation_from_matrix,translation_matrix
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Solver.point_line_registration import homo
from Solver.point_line_registration import LeastSquare_Solver as Solver
import yaml
import math 
from scipy.optimize import least_squares

with open('./data/fitting_set1.yaml') as stream:
    try:
        data = yaml.safe_load((stream))
    except yaml.YAMLERROIR as exc:
        print(exc)
data_list = data['Samples']
# print(data_list)

data_list = [d for i,d in zip(range(len(data_list)),data_list) if i not in [11,6]]
trus_spots = []
keys = ['TRUS1', 'TRUS2', 'TRUS3']# 'TRUS4']
thetas,us, vs = [],[],[]
for d in data_list:
    key = random.sample(keys,1)[0] # 'TRUS1'
    key = 'TRUS1'
    theta = d[key]['angle']
    u = d[key]['u']
    v = d[key]['v']
    y = -1*(0.01 + v ) * math.sin(theta/180.0 * math.pi)
    z = (0.01 + v ) * math.cos(theta/180.0 * math.pi)
    trus_spots.append([u, y, z])
    us.append(u)
    vs.append(v)
    thetas.append(theta)
trus_spots = np.array(trus_spots).T * 1000 # convert to mm


cam2marker_rotms = [quaternion_matrix(d['Marker']['quaternion']) for d in data_list]


cam2marker_translations = [d['Marker']['translation'] for d in data_list]
cam2marker_transforms = []
for i in range(len(cam2marker_rotms)):
    rotm = cam2marker_rotms[i]
    translation = cam2marker_translations[i]
    rotm[:3,3] = np.array(translation) * 1000
    # rotm = np.linalg.inv(rotm)
    cam2marker_transforms.append(rotm)
cam2marker_transforms = np.array(cam2marker_transforms)

direc_vec = unit_vector(-1.0 * np.array([-0.32871691, -0.10795516, -0.93823]))

cam_N = [unit_vector(cam2marker_transforms[i][:3,:3] @ direc_vec[:,None]) for i in range(len(cam2marker_transforms))]
cam_laser_start_spots = [cam2marker_transforms[i] @ np.array([ 49.9973,  18.979, -19.69427,1])[:,None] \
                             for i in range(len(cam2marker_transforms))]
# fitted results:  [ 0.08478048  0.03536605 -0.1202362 ] [-0.21044382 -0.05426389  0.97609878]
cam_N = np.concatenate(cam_N,axis=1)
cam_laser_start_spots = np.concatenate(cam_laser_start_spots,axis = 1)[:3]

idx =4
t  = thetas[idx]
u  = us[idx]
v = vs[idx]

trus_spot = np.array([u, -1*v*np.sin(t*np.pi/180), v*np.cos(t*np.pi/180)])*1000
cam_laser_start_spot = cam_laser_start_spots[:,idx] 
cam_N = cam_N[:,idx]
def fun(x,u,v,cam_laser_start_spots,cam_N,Freg):
    theta = x[:1]
    scale = x[1:]
    laser_spots_pred = cam_laser_start_spots + scale * cam_N
    trus_spot = np.array([u,-v * np.sin(theta*np.pi/180.0), v * np.cos(theta*np.pi/180.0)])[:,None] * 1000 # unit: mm
    diff = (Freg @ homo(trus_spot))[:3] - laser_spots_pred
    diff = np.array(diff)
    error = np.linalg.norm(diff) #+ 0.1 *np.sum( theta**2)
    print(error)
    return error


Freg = np.array([[-5.02140697e-01 , 8.51813669e-01 , 1.49225312e-01, -7.02880347e+00],
 [ 8.64380456e-01 , 4.89092791e-01 , 1.16767586e-01, -4.03049345e+01],
 [ 2.64792014e-02  ,1.87621201e-01, -9.81884483e-01,  2.22011082e+02],
 [ 0.00000000e+00  ,0.00000000e+00,  0.00000000e+00 , 1.00000000e+00]])

# Freg = np.array([[-4.95181165e-01 , 8.09645581e-01 , 3.15070860e-01 ,-1.68902134e+01],
#  [ 8.68106977e-01 , 4.75483086e-01,  1.42499514e-01 ,-4.17637519e+01],
#  [-3.44367627e-02 , 3.44078287e-01 ,-9.38309246e-01,  2.19955872e+02],
#  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

x = np.array([0,20])
result = least_squares(fun, x,\
                            bounds=([-30,0], [30,300] ),\
                                args=(u,v,cam_laser_start_spot[:,None],cam_N[:,None],Freg))
print(result.x)
print(t)


    
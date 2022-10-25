from functools import total_ordering
from turtle import color
import numpy as np
import random
import sys
sys.path.append('./')
from utils.transformations import identity_matrix,unit_vector,rotation_matrix, rotation_from_matrix,translation_matrix
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Solver.point_line_registration import homo
from Solver.point_line_registration import LeastSquare_Solver as Solver
from scipy.optimize import least_squares
import tqdm


# total number of pairs
line_num = 12
add_noise = True

# construct ground truth spots w.r.t. trus frame 
trus_bounding_box = np.asarray([40, 80, 15], np.float32) # x,y,z, unit: mm
# trus_spots = np.array([[-20,-40, 60]]).T+\
#     np.random.rand(3,line_num) * trus_bounding_box[:,None]
v_gt = 45 + np.random.rand(line_num) * 20
u_gt = -25 + np.random.rand(line_num) * 50
theta_gt = -30 + np.random.rand(line_num) * 60
trus_spots = np.concatenate([u_gt[None,:],        # x axis
                                -1*v_gt*np.sin(theta_gt*np.pi/180.0)[None,:],  # y axis
                                    v_gt*np.cos(theta_gt*np.pi/180.0)[None,:]], axis=0) # z axis
weight = 1.0 * (np.random.rand(line_num) > 0.5)
random_theta = (-6 + np.random.rand(line_num) * 3) * weight + (3 + np.random.rand(line_num) * 3 ) * (1 - weight)
rotms = [rotation_matrix(t*3.1415926/180.0,[1,0,0],[0,0,0]) for t in random_theta]
trus_spots_arc = [ rotm[:3,:3]@trus_spots[:,i][:,None]  \
                    for i, rotm in zip(range(line_num), rotms)]  
trus_spots_arc = np.concatenate(trus_spots_arc,axis=1)




# construct ground truth Freg (transformation from trus frame to camera frame) 
angle1 = (3.1415926/180.0)* (-100 -70 * random.random())
direction1 = np.array([0.15, 1,0.05],np.float32) 
angle2 = (3.1415926/180.0)* (-30 -60 * random.random())
direction2 = np.array([1, 0.1,0.2],np.float32) 
Freg = rotation_matrix(angle2, direction2) @ rotation_matrix(angle1, direction1)
Freg[:3,3] = np.array([-43,-190,270])
print(angle1/3.1415926 * 180,  angle2/3.1415926 * 180)       
print( Freg)     

# construct ground truth laser direction vectors w.r.t. camera frame 
trus_N = []
for i in range(line_num):
    trus_N.append(unit_vector(np.array([-2+10*random.random(),-2+10*random.random(),-10 - 20 *random.random()])))

trus_N =np.asarray(trus_N, dtype = np.float32).T
cam_N = Freg[:3,:3] @ trus_N
a = homo(trus_spots)
cam_spots = (Freg @ homo(trus_spots))[:-1,:]
trus_laser_start_spots = np.zeros_like(trus_spots)
x_gt = 15 + np.random.rand(line_num)*70.0
for i in range(line_num):
    trus_laser_start_spots[:,i] = trus_spots[:,i] + trus_N[:,i] * (-x_gt[i]) 
cam_laser_start_spots = (Freg @ homo(trus_laser_start_spots))[:-1,:]


theta = theta_gt + random_theta
idx =6
t  = theta[idx]
u  = u_gt[idx]
v = v_gt[idx]

trus_spot = np.array([u, -1*v*np.sin(t*np.pi/180), v*np.cos(t*np.pi/180)])
cam_laser_start_spot = cam_laser_start_spots[:,idx] 
cam_N = cam_N[:,idx]
def fun(x,u,v,t,cam_laser_start_spots,cam_N,Freg):
    theta = x[:1]
    scale = x[1:]
    laser_spots_pred = cam_laser_start_spots + scale * cam_N
    trus_spot = np.array([u,-v * np.sin((t+theta)*np.pi/180.0), v * np.cos((t+theta)*np.pi/180.0)])[:,None]  # unit: mm
    diff = (Freg @ homo(trus_spot))[:3] - laser_spots_pred
    diff = np.array(diff)
    error = np.linalg.norm(diff) #+ 0.1 *np.sum( theta**2)
    print(error)
    return error

x = np.array([0,20])
result = least_squares(fun, x,\
                            bounds=([-6,0], [6,300] ),\
                                args=(u,v,t,cam_laser_start_spot[:,None],cam_N[:,None],Freg))
print(result.x[0]+t, result.x[1])
print(theta_gt[idx],x_gt[idx])


    
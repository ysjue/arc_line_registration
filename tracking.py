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
import os

# angle u v intensity
root = './data/sample1.txt'
txt_files = os.listdir(root)
for txt_file in txt_files:
    txt_file = os.path.join(root, txt_file)
    with open(txt_file, 'r') as file:
        contents = file.readlines()
    contents = [c.split('/n')[0] for c in contents]
    samples = np.array(contents)
    gt_idx = np.argsort(samples[:,3])[-1]

idxes = [i for i in range(samples.shape[1])]
trus_spots = []
for i in range(samples.shape[1]):
    idx = random.sample(idxes,1)[0]
    theta = samples[idx, 0]
    u = samples[idx, 1]
    v = samples[idx, 2]
    y = -1*(0.01 + v ) * math.sin(theta/180.0 * math.pi)
    z = (0.01 + v ) * math.cos(theta/180.0 * math.pi)
    trus_spots.append([u, y, z])
trus_spots = np.array(trus_spots).T * 1000 # convert to mm

cam2marker_rotm = quaternion_matrix(np.array([0.988, -0.048, 0.039, -0.143])) 


cam2marker_translation = np.array([-0.016, 0.016, 0.090]) * 1000 # unit: mm
cam2marker_transforms = []

rotm = cam2marker_rotm
translation = cam2marker_translation
rotm[:3,3] = translation
cam2marker_transforms = rotm


direc_vec = unit_vector(-1.0 * np.array([-0.32871691, -0.10795516, -0.93823]))

cam_N = unit_vector(cam2marker_transforms[:3,:3] @ direc_vec[:,None]) 
cam_laser_start_spot = cam2marker_transforms @ np.array([ 49.9973,  18.979, -19.69427,1])[:,None] 
                           
# fitted results:  [ 0.08478048  0.03536605 -0.1202362 ] [-0.21044382 -0.05426389  0.97609878]


# visualization
colors = ['b','g','r','c','m','y','k','brown','gold','teal','plum']
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('X Label (mm)')
ax.set_ylabel('Y Label (mm)')
ax.set_zlabel('Z Label (mm)')


def fun(theta,trus_spots,laser_spots_pred,Freg):
    rotms = [rotation_matrix(t*3.1415926/180.0,[1,0,0],[0,0,0]) for t in theta]
    trus_spots = [ rotm[:3,:3] @ trus_spots[:,i][:,None]  \
                    for i, rotm in zip(range(len(theta)), rotms)] 
    trus_points = np.concatenate(trus_spots, axis = 1)
    
    error = np.sum(np.linalg.norm((Freg @ homo(trus_points))[:3] - laser_spots_pred, axis = 0)) #+ 0.1 *np.sum( theta**2)
    return error
error2 = 1000
lst_error2 = 10000
F2 =identity_matrix()


theta = np.zeros(trus_spots.shape[1])
result = least_squares(fun, theta,\
                        bounds=([-5.0]*trus_spots.shape[1], [5.0] * trus_spots.shape[1]),\
                            args=(trus_spots, cam_spots_pred, F))
thetas = result.x
print('estimated rotating angles: ', -1 * thetas)

    
F2=lst_F2
print('error1: ', error)

for i in range(trus_spots_pred2.shape[1]):
    ax.scatter(trus_laser_start_spots[0,i], trus_laser_start_spots[1,i], trus_laser_start_spots[2,i], marker='*',color=colors[i if i < 11 else 10])

    ax.scatter(trus_spots_pred2[0,i], trus_spots_pred2[1,i], trus_spots_pred2[2,i], marker='^',color=colors[i if i < 11 else 10])
    ax.scatter(trus_spots[0,i], trus_spots[1,i], trus_spots[2,i], marker='o',color=colors[i if i < 11 else 10])
    ax.scatter(trus_spots_pred1[0,i], trus_spots_pred1[1,i], trus_spots_pred1[2,i], marker='x',color=colors[i if i < 11 else 10])
    # trus_N = F[:3,:3].T@cam_N
    
    # ax.quiver(trus_laser_start_spots[0,i], trus_laser_start_spots[1,i], trus_laser_start_spots[2,i],\
    #             trus_N[0,i],trus_N[1,i], trus_N[2,i],\
    #             length = x_pred[i],color=colors[i if i < 11 else 10])

plt.show()

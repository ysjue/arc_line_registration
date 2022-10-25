#%%
import math
import os
import random

import numpy as np
import yaml
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares

from Solver.point_line_registration import LeastSquare_Solver as Solver
from Solver.point_line_registration import homo
from utils.transformations import (identity_matrix, quaternion_matrix,
                                   rotation_from_matrix, rotation_matrix,
                                   translation_matrix, unit_vector)

file_path = './data/testset1'
trus_samples_txt = [f for f in os.listdir(file_path) if 'testset' in f and 'cam' not in f]
cam_txt = os.path.join(file_path, 'testset_cam.txt')
samples = []
data_dict = {'Samples':[]}
cams = []
with open(cam_txt,'r') as f:
    lines = f.readlines()
    lines = [l.split('\n')[0] for l in lines if l != '\n' \
                        and 'Rotation' not in l and 'Translation' not in l]
cam_samples = []
for line in lines:
    line = line.split(']')[0].split('[')[1]
    line = [ float(l) for l in line.split(', ')]
    cam_samples.append(line)
# print(cam_samples)
cam2marker_transforms = []
for i in range(int(len(cam_samples)/2)):
    translation = np.array(cam_samples[i])
    rotm = quaternion_matrix(cam_samples[int(len(cam_samples)/2+ i)])
    rotm[:3,3] = translation*1000
    cam2marker_transforms.append(rotm)


for sample_txt in trus_samples_txt:
    sample_txt = os.path.join(file_path, sample_txt)
    sample = []
    with open(sample_txt,'r') as f:
        lines = f.readlines()
        # lines = [l.split('\n')[0] for l in lines]

    for line in lines:
        line = [float(l) for l in line.split(' ') if l != '' ]
        sample.append(line)
        
    sample = np.array(sample)
    samples.append(sample)

samples = samples[:13]+samples[14:]
direc_vec = unit_vector( np.array([ 0.03135577,  0.21365248, -0.97640639]))
cam2marker_transforms = cam2marker_transforms[:13]+cam2marker_transforms[14:]
cam_N = [unit_vector(cam2marker_transforms[i][:3,:3] @ direc_vec[:,None]) for i in range(len(cam2marker_transforms))]
cam_laser_start_spots = [cam2marker_transforms[i] @ np.array([37.64178, -14.98675, -2.07052,1])[:,None] \
                             for i in range(len(cam2marker_transforms))]
# fitted results:  [ 0.08478048  0.03536605 -0.1202362 ] [-0.21044382 -0.05426389  0.97609878]
cam_N = np.concatenate(cam_N,axis=1)
cam_laser_start_spots = np.concatenate(cam_laser_start_spots,axis = 1)[:3]


acc_sample = [[sample[0,:][np.argmax(sample[-1,:])], sample[1,:][np.argmax(sample[-1,:])],\
                    sample[2,:][np.argmax(sample[-1,:])],sample[3,:][np.argmax(sample[-1,:])]] for sample in samples]
acc_sample = np.array(acc_sample)
acc_theta = acc_sample[:,0]
acc_u = acc_sample[:,1] * 1000 # unit: mm
acc_v = acc_sample[:,2] * 1000 # unit: mm
acc_intensity= acc_sample[:,-1]

# deviation = int(-5 + np.random.rand()*10)
# sample = [[sample[0,:][np.argmax(sample[-1,:])+deviation], sample[1,:][np.argmax(sample[-1,:])+deviation],\
#                     sample[2,:][np.argmax(sample[-1,:])+deviation]] for sample in samples]
# sample = np.array(sample)

trus_spots = np.concatenate([acc_u[None,:],        # x axis
                                    -1*(10+acc_v)*np.sin(acc_theta*np.pi/180.0)[None,:],  # y axis
                                        (10+acc_v)*np.cos(acc_theta*np.pi/180.0)[None,:]], axis=0) # z axis
# trus_spot = np.array([u, -1*v*np.sin(t*np.pi/180), v*np.cos(t*np.pi/180)])*1000

# visualization
colors = ['b','g','r','c','m','y','k','brown','gold','teal','plum']
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('X Label (mm)')
ax.set_ylabel('Y Label (mm)')
ax.set_zlabel('Z Label (mm)')

solver1 = Solver(geo_consist=False)

F1,error,lower,upper = solver1.solve(trus_spots, cam_laser_start_spots,cam_N, F0=identity_matrix())
x_pred, cam_spots_pred1 = solver1.output()

trus_spots_arc = trus_spots
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
cam_spots_pred = cam_spots_pred1*1
count = 0
F = F1
while error2 < lst_error2: 
    count+=1
    theta = np.zeros(trus_spots.shape[1])
    result = least_squares(fun, theta,\
                            bounds=([-5.0]*trus_spots.shape[1], [5.0] * trus_spots.shape[1] ),\
                                args=(trus_spots_arc, cam_spots_pred,F))
    thetas = result.x
    print('estimated rotating angles: ', -1 * thetas)

    
    # update trus_spots_arc
    trus_spots_arc = [rotation_matrix(theta*np.pi/180.0,[1,0,0],[0,0,0])[:3,:3] @ trus_spots_arc[:,i][:,None] \
                                    for i, theta in zip(range(trus_spots.shape[1]), thetas)]
    trus_spots_arc = np.concatenate(trus_spots_arc, axis=1)
    lst_error2 = error2
    lst_F2 = F2
    solver2 = Solver(geo_consist=False)
    F2,error2,lower2,upper2 = solver2.solve(trus_spots_arc, cam_laser_start_spots,cam_N, F0=F)
    print('error2: ',error2 )
    F = F2
    x_pred2, cam_spots_pred = solver2.output()
    if error2 < 0.02 or count > 50:
        print("slight rotation")
        break
F2=lst_F2
print('error1: ', error,lower,upper)
print('error2: ', error2,lower2,upper2)
print(F2)

trus_laser_start_spots = np.linalg.inv(F2)[:3,:3] @ cam_laser_start_spots + np.linalg.inv(F2)[:3,3][:,None]
trus_spots_pred2 = (np.linalg.inv(F2) @ homo(cam_spots_pred))[:-1,:]
trus_spots_pred1 = (np.linalg.inv(F1) @ homo(cam_spots_pred1))[:-1,:]

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


# %%

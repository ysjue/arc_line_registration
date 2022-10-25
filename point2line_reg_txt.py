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

file_path = './data/testset1'
trus_samples_txt = [f for f in os.listdir(file_path) if 'sample' in f and 'cam' not in f]
cam_txt = os.path.join(file_path, 'sample_cam.txt')
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

# samples = [sample for i, sample in zip(range(len(samples)), samples) if i not in [11,  2,  7,  4, 12,  6, 13,  9,  1]]
# cam2marker_transforms = [transform for i, transform in zip(range(len(cam2marker_transforms)),\
#                      cam2marker_transforms)  if i not in [11,  2 , 7  ,4, 12 , 6, 13,  9,  1]]
direc_vec = unit_vector( np.array([-0.32871691, -0.10795516, -0.93823]))

cam_N = [unit_vector(cam2marker_transforms[i][:3,:3] @ direc_vec[:,None]) for i in range(len(cam2marker_transforms))]
cam_laser_start_spots = [cam2marker_transforms[i] @ np.array([ 49.9973,  18.979, -19.69427,1])[:,None] \
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
                                    -(acc_v+10)*np.sin(acc_theta*np.pi/180.0)[None,:],  # y axis
                                        (acc_v+10)*np.cos(acc_theta*np.pi/180.0)[None,:]], axis=0) # z axis
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
print(error,x_pred)
trus_laser_start_spots = np.linalg.inv(F1)[:3,:3] @ cam_laser_start_spots + np.linalg.inv(F1)[:3,3][:,None]

trus_spots_pred1 = (np.linalg.inv(F1) @ homo(cam_spots_pred1))[:-1,:]

for i in range(trus_spots_pred1.shape[1]):
    ax.scatter(trus_laser_start_spots[0,i], trus_laser_start_spots[1,i], trus_laser_start_spots[2,i], marker='*',color=colors[i if i < 11 else 10])

    ax.scatter(trus_spots[0,i], trus_spots[1,i], trus_spots[2,i], marker='o',color=colors[i if i < 11 else 10])
    # ax.scatter(trus_spots_pred1[0,i], trus_spots_pred1[1,i], trus_spots_pred1[2,i], marker='x',color=colors[i if i < 11 else 10])
    # trus_N = F[:3,:3].T@cam_N
    
    # ax.quiver(trus_laser_start_spots[0,i], trus_laser_start_spots[1,i], trus_laser_start_spots[2,i],\
    #             trus_N[0,i],trus_N[1,i], trus_N[2,i],\
    #             length = x_pred[i],color=colors[i if i < 11 else 10])

plt.show()
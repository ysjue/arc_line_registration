import os
import numpy as np
import scipy.stats as st
import yaml
from utils.transformations import rotation_matrix, quaternion_matrix, unit_vector
from Solver.point_line_registration import homo
from scipy.optimize import least_squares

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
    sample_dict = []
    samples.append(sample)
# print(samples)

direc_vec = unit_vector(np.array([-0.32871691, -0.10795516, -0.93823]))

cam_Ns = [unit_vector(cam2marker_transforms[i][:3,:3] @ direc_vec[:,None]) for i in range(len(cam2marker_transforms))]
cam_laser_start_spots = [cam2marker_transforms[i] @ np.array([ 49.9973,  18.979, -19.69427,1])[:,None] \
                             for i in range(len(cam2marker_transforms))]
# fitted results:  [ 0.08478048  0.03536605 -0.1202362 ] [-0.21044382 -0.05426389  0.97609878]
cam_Ns = np.concatenate(cam_Ns,axis=1)
cam_laser_start_spots = np.concatenate(cam_laser_start_spots,axis = 1)[:3]

sample = [[sample[0,:][np.argmax(sample[-1,:])], sample[1,:][np.argmax(sample[-1,:])],\
                     sample[2,:][np.argmax(sample[-1,:])]] for sample in samples]
sample = np.array(sample)

idx =2
t  = sample[idx,0]
u  = sample[idx,1]
v = sample[idx,2]

trus_spot = np.array([u, -1*v*np.sin(t*np.pi/180), v*np.cos(t*np.pi/180)])*1000
cam_laser_start_spot = cam_laser_start_spots[:,idx] 
cam_N = cam_Ns[:,idx]
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

Freg = np.array([[-4.95181165e-01 , 8.09645581e-01 , 3.15070860e-01 ,-1.68902134e+01],
 [ 8.68106977e-01 , 4.75483086e-01,  1.42499514e-01 ,-4.17637519e+01],
 [-3.44367627e-02 , 3.44078287e-01 ,-9.38309246e-01,  2.19955872e+02],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

x = np.array([0,20])
result = least_squares(fun, x,\
                            bounds=([-30,0], [30,300] ),\
                                args=(u,v,cam_laser_start_spot[:,None],cam_N[:,None],Freg))
print(result.x)
print(t)


    
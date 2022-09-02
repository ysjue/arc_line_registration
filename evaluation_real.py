import os
import numpy as np
import scipy.stats as st
import yaml
from utils.transformations import quaternion_matrix, unit_vector

file_path = './data/testset1'
trus_samples_txt = [f for f in os.listdir(file_path) if 'testset' in f]
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
print(cam_samples)
cam2marker_transforms = []
for i in range(len(cam_samples)/2):
    translation = np.array(cam_samples[i])
    rotm = quaternion_matrix(cam_samples[len(cam_samples)/2+ i])
    rotm[:3,3] = translation
    cam2marker_transforms.append(rotm)


for sample_txt in trus_samples_txt:
    sample_txt = os.path.join(file_path, sample_txt)
    sample = []
    with open(sample_txt,'r') as f:
        lines = f.readlines()
        lines = [l.split('\n')[0] for l in lines]
    for line in lines:
        line = [float(l) for l in line.split(' ') if l != '' ]
        sample.append(line)
    sample = np.array(sample)
    sample_dict = []
    samples.append(sample)
    

direc_vec = unit_vector(-1.0 * np.array([-0.32871691, -0.10795516, -0.93823]))

cam_N = [unit_vector(cam2marker_transforms[i][:3,:3] @ direc_vec[:,None]) for i in range(len(cam2marker_transforms))]
cam_laser_start_spots = [cam2marker_transforms[i] @ np.array([ 49.9973,  18.979, -19.69427,1])[:,None] \
                             for i in range(len(cam2marker_transforms))]
# fitted results:  [ 0.08478048  0.03536605 -0.1202362 ] [-0.21044382 -0.05426389  0.97609878]
cam_N = np.concatenate(cam_N,axis=1)
cam_laser_start_spots = np.concatenate(cam_laser_start_spots,axis = 1)[:3]





    
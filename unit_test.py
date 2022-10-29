import os
import yaml
import numpy as np



root = '/home/sean/arc_line_reg/data'
file = os.path.join(root,'sample1.txt')
with open(file,'r') as f:
  content = f.readlines()
samples = [c.split('\n')[0] for c in content]
# samples = np.array(samples)
data = {'Samples':[]}
intensity = samples[-1]
intensities = np.array([float(i) for i in intensity.split(' ') if i != ''])
theta = samples[0]
thetas = np.array([float(i) for i in theta.split(' ') if i != ''])
u = samples[-1]
us = np.array([float(i) for i in u.split(' ') if i != ''])
v = samples[-1]
vs = np.array([float(i) for i in v.split(' ') if i != ''])
sample = np.concatenate([thetas[:,None],us[:,None],vs[:,None],\
            intensities[:,None]], axis=1)
gt_idx = np.argsort(sample[:,-1])
gt_u = sample[gt_idx]

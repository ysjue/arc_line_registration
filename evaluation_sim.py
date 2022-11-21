import os
import numpy as np
import scipy.stats as st

file_path = './results/simulation_with_slight_noise_40.txt'
with open(file_path,'r') as f:
    lines = f.readlines()
samples = []
for line in lines:
    line = line.split('\n')[0]
    if 'noise' in line:
        continue
    vals = [float(val_str) for val_str in line.split(' ') if val_str != '']
    print(vals)
    assert len(vals) == 12
    
    samples.append(vals)
samples = np.array(samples)
results = []
for i in range(samples.shape[1]):
    lower, upper = st.t.interval(alpha=0.95, df=len(samples[:,i])-1, 
                                    loc=np.mean(samples[:,i]), 
                                    scale=st.sem(samples[:,i])) 
    mean_val = np.mean(samples[:,i])
    results.append([mean_val,np.std(samples[:,i]),lower,upper])
print(np.array(results[-6:]))
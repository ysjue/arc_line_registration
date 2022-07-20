#%%
from turtle import color
import numpy as np
import random
from transformations import identity_matrix,unit_vector,rotation_matrix, rotation_from_matrix,translation_matrix
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from arc_line_registration import homo, Solver

line_num = 10
trus_bounding_box = np.asarray([-160, 100, 20], np.float32) # x,y,z, unit: mm
trus_spots = np.array([[-50,-50, 0]]).T+\
    np.random.rand(3,line_num) * trus_bounding_box[:,None]
noises = np.random.rand(trus_spots.shape[0], trus_spots.shape[1])*15.0

angle1 = (3.1415926/180.0)* (-100 -70 * random.random())
direction1 = np.array([0.15, 1,0.05],np.float32) 
angle2 = (3.1415926/180.0)* (-30 -60 * random.random())
direction2 = np.array([1, 0.1,0.2],np.float32) 
Freg = rotation_matrix(angle2, direction2) @ rotation_matrix(angle1, direction1)
Freg[:3,3] = np.array([-43,-190,520])
print(angle1/3.1415926 * 180,  angle2/3.1415926 * 180)       
print( Freg)     

trus_N = []
for i in range(line_num):
    trus_N.append(unit_vector(np.array([-5+10*random.random(),-5+10*random.random(),-10 + -20 *random.random()])))
# trus_N = [unit_vector(np.array([-5+10*random.random(),1,-4])),\
#     unit_vector(np.array([1,2.3,-3.5])),\
#     unit_vector(np.array([-1,0,-4])),\
#     unit_vector(np.array([0.5,-2,-6])),\
#     unit_vector(np.array([-2,1,-7])),\
#     unit_vector(np.array([3,-1,-6]))]
trus_N =np.asarray(trus_N, dtype = np.float32).T
cam_N = Freg[:3,:3] @ trus_N
a = homo(trus_spots)
cam_spots = (Freg @ homo(trus_spots))[:-1,:]
trus_laser_start_spots = np.zeros_like(trus_spots)
x = 15 + np.random.rand(line_num)*70.0
for i in range(line_num):
    trus_laser_start_spots[:,i] = trus_spots[:,i] + trus_N[:,i] * (-x[i]) 

cam_laser_start_spots = (Freg @ homo(trus_laser_start_spots))[:-1,:]

# visualization
colors = ['b','g','r','c','m','y','k','brown','gold','teal','plum']
fig = plt.figure()
ax = fig.gca(projection='3d')
for i in range(trus_spots.shape[1]):
    ax.scatter(trus_spots[0,i], trus_spots[1,i], trus_spots[2,i], marker='o',color=colors[i if i < 11 else 10])
    ax.scatter(trus_laser_start_spots[0,i], trus_laser_start_spots[1,i], \
                trus_laser_start_spots[2,i], marker='^',color=colors[i if i < 11 else 10])
    ax.quiver(trus_laser_start_spots[0,i], trus_laser_start_spots[1,i], trus_laser_start_spots[2,i],\
                trus_N[0,i],trus_N[1,i], trus_N[2,i],\
                length = x[i],color=colors[i if i < 11 else 10])
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

solver1 = Solver(geo_consist=False)
solver2 = Solver(geo_consist=True)

F,error = solver1.solve(trus_spots+noises, cam_laser_start_spots,cam_N, F0=identity_matrix())
x_pred, cam_spots_pred = solver1.output()
_,error2 = solver2.solve(trus_spots+noises, cam_laser_start_spots,cam_N, F0=identity_matrix())
x_pred2, cam_spots_pred2 = solver2.output()
print(error,error2)
print('x_pred1:', x_pred[:,0])
print('x_pred2:', x_pred2[:,0])
print('x gt:', x)
trus_spots_pred = (np.linalg.inv(F) @ homo(cam_spots_pred))[:-1,:]
for i in range(trus_spots_pred.shape[1]):
    ax.scatter(trus_spots_pred[0,i], trus_spots_pred[1,i], trus_spots_pred[2,i], marker='*',color=colors[i if i < 11 else 10])
plt.show()

# %%

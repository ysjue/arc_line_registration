#%%
import numpy as np
import random
from utils.transformations import identity_matrix,unit_vector,rotation_matrix,quaternion_matrix, rotation_from_matrix,translation_matrix
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Solver.point_line_registration import homo
from Solver.point_line_registration import LeastSquare_Solver as Solver
import yaml
import math 

with open('./data/point2line_data.yaml') as stream:
    try:
        data = yaml.safe_load((stream))
    except yaml.YAMLERROIR as exc:
        print(exc)
data_list = data['Samples']
trus_spots = []
for d in data_list:
    theta = d['TRUS1']['angle']
    u = d['TRUS1']['u']
    v = d['TRUS1']['v']
    y = -1*(0.01 + v ) * math.sin(theta/180.0 * math.pi)
    z = (0.01 + v ) * math.cos(theta/180.0 * math.pi)
    trus_spots.append([u, y, z])
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

direc_vec = unit_vector(-1.0 * np.array([-0.24993895, -0.04863153,  0.96703955]))

cam_N = [unit_vector(cam2marker_transforms[i][:3,:3] @ direc_vec[:,None]) for i in range(len(cam2marker_transforms))]
cam_laser_start_spots = [cam2marker_transforms[i] @ np.array([58.76608, 27.99237, 13.31244,1])[:,None] \
                             for i in range(len(cam2marker_transforms))]
# fitted results:  [ 0.08478048  0.03536605 -0.1202362 ] [-0.21044382 -0.05426389  0.97609878]
cam_N = np.concatenate(cam_N,axis=1)
cam_laser_start_spots = np.concatenate(cam_laser_start_spots,axis = 1)[:3]


# visualization
colors = ['b','g','r','c','m','y','k','brown','gold','teal','plum']
fig = plt.figure()
ax = fig.gca(projection='3d')
# for i in range(trus_spots.shape[1]):
#     ax.scatter(trus_spots[0,i], trus_spots[1,i], trus_spots[2,i], marker='o',color=colors[i if i < 11 else 10])
#     ax.scatter(trus_laser_start_spots[0,i], trus_laser_start_spots[1,i], \
#                 trus_laser_start_spots[2,i], marker='^',color=colors[i if i < 11 else 10])
#     ax.quiver(trus_laser_start_spots[0,i], trus_laser_start_spots[1,i], trus_laser_start_spots[2,i],\
#                 trus_N[0,i],trus_N[1,i], trus_N[2,i],\
#                 length = x[i],color=colors[i if i < 11 else 10])
ax.set_xlabel('X Label (mm)')
ax.set_ylabel('Y Label (mm)')
ax.set_zlabel('Z Label (mm)')

solver1 = Solver(geo_consist=False)
solver2 = Solver(geo_consist=True)
F0 = np.array([[-3.41611997e-01 , 3.82063185e-01, -8.58678616e-01, -4.30000000e+01],
 [ 7.17263294e-01,  6.96371615e-01,  2.44936933e-02, -1.90000000e+02],
 [ 6.07317554e-01, -6.07531313e-01 ,-5.11928797e-01,  5.20000000e+02],
 [ 0.00000000e+00 , 0.00000000e+00  ,0.00000000e+00 , 1.00000000e+00]])
F,error,lower,upper = solver1.solve(trus_spots, cam_laser_start_spots,cam_N, F0=identity_matrix())
x_pred, cam_spots_pred = solver1.output()
print('error1: ', error,lower,upper)
print('x_pred1:', x_pred[:,0])
print(np.linalg.inv(F))

trus_laser_start_spots = np.linalg.inv(F)[:3,:3] @ cam_laser_start_spots + np.linalg.inv(F)[:3,3][:,None]
trus_spots_pred = (np.linalg.inv(F) @ homo(cam_spots_pred))[:-1,:]

for i in range(trus_spots_pred.shape[1]):
    ax.scatter(trus_laser_start_spots[0,i], trus_laser_start_spots[1,i], trus_laser_start_spots[2,i], marker='*',color=colors[i if i < 11 else 10])

    ax.scatter(trus_spots_pred[0,i], trus_spots_pred[1,i], trus_spots_pred[2,i], marker='^',color=colors[i if i < 11 else 10])
    ax.scatter(trus_spots[0,i], trus_spots[1,i], trus_spots[2,i], marker='o',color=colors[i if i < 11 else 10])
    # trus_N = F[:3,:3].T@cam_N
    
    # ax.quiver(trus_laser_start_spots[0,i], trus_laser_start_spots[1,i], trus_laser_start_spots[2,i],\
    #             trus_N[0,i],trus_N[1,i], trus_N[2,i],\
    #             length = x_pred[i],color=colors[i if i < 11 else 10])

plt.show()


# %%

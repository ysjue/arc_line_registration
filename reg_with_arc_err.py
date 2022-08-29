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
from scipy.optimize import least_squares

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
    if error2 < 0.02 or count > 55:
        print("slight rotation")
        break
F2=lst_F2
print('error1: ', error,lower,upper)
print('error2: ', error2,lower2,upper2)
print(np.linalg.inv(F2))

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

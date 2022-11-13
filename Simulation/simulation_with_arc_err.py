from turtle import color
import numpy as np
import random
import sys
sys.path.append('./')
from utils.transformations import identity_matrix,unit_vector,rotation_matrix, rotation_from_matrix,translation_matrix
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Solver.point_line_registration import homo
from Solver.point_line_registration import LeastSquare_Solver as Solver
from scipy.optimize import least_squares



# total number of pairs
line_num = 10
add_noise = True

# construct ground truth spots w.r.t. trus frame 
trus_bounding_box = np.asarray([40, 80, 15], np.float32) # x,y,z, unit: mm
trus_spots = np.array([[-20,-40, 60]]).T+\
    np.random.rand(3,line_num) * trus_bounding_box[:,None]
trus_spots_noises = np.ones_like(trus_spots) * np.array([-1,-1,-2])[:,None]+\
                            np.random.rand(trus_spots.shape[0], trus_spots.shape[1])*np.array([2,2.0,4.0])[:,None]
weight = 1.0 * (np.random.rand(line_num) > 0.5)
random_theta = (-6 + np.random.rand(line_num) * 3) * weight + (2 + np.random.rand(line_num) * 4 ) * (1 - weight)
rotms = [rotation_matrix(theta*3.1415926/180.0,[1,0,0],[0,0,0]) for theta in random_theta]
trus_spots_arc = [ rotm[:3,:3]@trus_spots[:,i][:,None]  \
                    for i, rotm in zip(range(line_num), rotms)]  
trus_spots_arc = np.concatenate(trus_spots_arc,axis=1)




# construct ground truth Freg (transformation from trus frame to camera frame) 
angle1 = (3.1415926/180.0)* (-100 -70 * random.random())
direction1 = np.array([0.15, 1,0.05],np.float32) 
angle2 = (3.1415926/180.0)* (-30 -60 * random.random())
direction2 = np.array([1, 0.1,0.2],np.float32) 
Freg = rotation_matrix(angle2, direction2) @ rotation_matrix(angle1, direction1)
Freg[:3,3] = np.array([-43,-190,270])
print(angle1/3.1415926 * 180,  angle2/3.1415926 * 180)       
print( Freg)     

# construct ground truth laser direction vectors w.r.t. camera frame 
trus_N = []
for i in range(line_num):
    trus_N.append(unit_vector(np.array([-2+10*random.random(),-2+10*random.random(),-10 - 20 *random.random()])))

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

solver0 = Solver(geo_consist=False)
solver1 = Solver(geo_consist=False)

# calculate the solution
if add_noise:
    trus_spots_arc = trus_spots_arc + trus_spots_noises
    trus_spots     = trus_spots + trus_spots_noises
    for i in range(cam_N.shape[1]):
        cam_N[:,i] = unit_vector(cam_N[:,i] + np.array([0.02 * random.random(), 0.02 * random.random(), -0.1 * random.random()]))
F0,error0,lower0,upper0 = solver0.solve(trus_spots, cam_laser_start_spots, cam_N, F0=identity_matrix())
F,error1,lower,upper = solver1.solve(trus_spots_arc, cam_laser_start_spots, cam_N, F0=identity_matrix())
F1 = F*1
x_pred1, cam_spots_pred1 = solver1.output()
#============ arc-correction================
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
cam_spots_pred = cam_spots_pred1
count = 0
trus_spots_arc1 = trus_spots_arc
error2s = [error1]
while count<55:#error2 < lst_error2: 
    count+=1
    theta = np.zeros(line_num)
    result = least_squares(fun, theta,\
                            bounds=([-5.0]*line_num, [5.0] * line_num ),\
                                args=(trus_spots_arc, cam_spots_pred,F))
    thetas = result.x
    
    print('estimated rotating angles: ', -1 * thetas)
    print('real rotating angles: ', random_theta)
    
    # update trus_spots_arc
    trus_spots_arc = [rotation_matrix(theta*np.pi/180.0,[1,0,0],[0,0,0])[:3,:3] @ trus_spots_arc[:,i][:,None] \
                                    for i, theta in zip(range(line_num), thetas)]
    trus_spots_arc = np.concatenate(trus_spots_arc, axis=1)
    lst_error2 = error2
    lst_F2 = F2
    solver2 = Solver(geo_consist=False)
    F2,error2,lower2,upper2 = solver2.solve(trus_spots_arc, cam_laser_start_spots,cam_N, F0=F)
    print('error2: ',error2 )
    error2s.append(error2)
    F = F2
    x_pred2, cam_spots_pred = solver2.output()
    if error2 < 2e-5 or count > 200:
        print("slight rotation")
        break
F2=lst_F2
print(lst_error2,error1, error0)
print('x_pred1:', x_pred1[:,0])
print('x_pred2:', x_pred2[:,0])
print('x gt:', x)
print('translation error0:', np.linalg.norm(F0[:3,3] - Freg[:3,3]))
print('translation error1:', np.linalg.norm(F1[:3,3] - Freg[:3,3]))
print('translation error2:', np.linalg.norm(F2[:3,3] - Freg[:3,3]))
print(error2s)


homo_matrix0, homo_matrix1, homo_matrix2= np.zeros((4,4)),np.zeros((4,4)), np.zeros((4,4))
homo_matrix0[3,3] = 1
homo_matrix1[3,3] = 1
homo_matrix2[3,3] = 1
homo_matrix0[:3,:3] = F0[:3,:3] @ Freg[:3,:3].T
homo_matrix1[:3,:3] = F1[:3,:3] @ Freg[:3,:3].T
homo_matrix2[:3,:3] = F2[:3,:3] @ Freg[:3,:3].T
angle0,_,_ = rotation_from_matrix(homo_matrix0 )
angle1,_,_ = rotation_from_matrix(homo_matrix1 )
angle2,_,_=rotation_from_matrix(homo_matrix2 )
print('rotation error0: ', angle0*180.0 /np.pi)
print('rotation error1: ', angle1*180.0 /np.pi )
print('rotation error2: ', angle2*180.0 /np.pi)
trus_spots_pred1 = (np.linalg.inv(F1) @ homo(cam_spots_pred1))[:-1,:]
trus_spots_pred2 = (np.linalg.inv(F2) @ homo(cam_spots_pred))[:-1,:]
for i in range(trus_spots_pred2.shape[1]):
    ax.scatter(trus_spots_pred2[0,i], trus_spots_pred2[1,i], trus_spots_pred2[2,i], marker='*',color=colors[i if i < 11 else 10])
    ax.scatter(trus_spots_pred1[0,i], trus_spots_pred1[1,i], trus_spots_pred1[2,i], marker='x',color=colors[i if i < 11 else 10])
plt.show()


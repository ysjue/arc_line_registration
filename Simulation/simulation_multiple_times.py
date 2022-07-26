import random
import sys
from functools import total_ordering
from turtle import color

import numpy as np

sys.path.append('./')
import tqdm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares

from Solver.point_line_registration import LeastSquare_Solver as Solver
from Solver.point_line_registration import homo
from utils.transformations import (identity_matrix, rotation_from_matrix,
                                   rotation_matrix, translation_matrix,
                                   unit_vector)


def main():
    # total number of pairs
    line_num = 20
    add_noise = True

    # construct ground truth spots w.r.t. trus frame 
    trus_bounding_box = np.asarray([40, 80, 10], np.float32) # x,y,z, unit: mm
    # trus_spots = np.array([[-20,-40, 60]]).T+\
    #     np.random.rand(3,line_num) * trus_bounding_box[:,None]
    v_gt = 55 + np.random.rand(line_num) * 7
    u_gt = -25 + np.random.rand(line_num) * 50
    theta_gt = -18 + np.random.rand(line_num) * 36
    trus_spots = np.concatenate([u_gt[None,:],        # x axis
                                    -1*v_gt*np.sin(theta_gt*np.pi/180.0)[None,:],  # y axis
                                        v_gt*np.cos(theta_gt*np.pi/180.0)[None,:]], axis=0) # z axis
  
    weight = 1.0 * (np.random.rand(line_num) > 0.5)
    random_theta = (-6 + np.random.rand(line_num) * 3) * weight + (3 + np.random.rand(line_num) * 3) * (1 - weight)
    rotms = [rotation_matrix(t*3.1415926/180.0,[1,0,0],[0,0,0]) for t in random_theta]
    trus_spots_arc = [ rotm[:3,:3]@trus_spots[:,i][:,None]  \
                        for i, rotm in zip(range(line_num), rotms)]  
    trus_spots_arc = np.concatenate(trus_spots_arc,axis=1)




    # construct ground truth Freg (transformation from trus frame to camera frame) 
    angle1 = (3.1415926/180.0)* (-100 -70 * random.random())
    direction1 = np.array([0.15, 1,0.05],np.float32) 
    angle2 = (3.1415926/180.0)* (-30 -60 * random.random())
    direction2 = np.array([1, 0.1,0.2],np.float32) 
    Freg = rotation_matrix(angle2, direction2) @ rotation_matrix(angle1, direction1)
    Freg[:3,3] = np.array([-43,-390,300])
    print(angle1/3.1415926 * 180,  angle2/3.1415926 * 180)       
    Freg= np.array([[-4.95565366e-01,  8.42537551e-01 , 2.11057912e-01, -1.07281469e+01],
    [ 8.68272044e-01 , 4.86918007e-01 , 9.49447830e-02 ,-3.89869408e+01],
    [-2.27733530e-02,  2.30307031e-01, -9.72851503e-01,  2.91642210e+02],
    [ 0.00000000e+00,  0.00000000e+00 , 0.00000000e+00 , 1.00000000e+00]])
    print( Freg)     

    # construct ground truth laser direction vectors w.r.t. camera frame 
    trus_N = []
    for i in range(line_num):
        trus_N.append(unit_vector(np.array([-2+10*random.random(),-2+10*random.random(),-10 - 20 *random.random()])))

    trus_N =np.asarray(trus_N, dtype = np.float32).T
    cam_N = Freg[:3,:3] @ trus_N
    a = homo(trus_spots)
    cam_spots = (Freg @ homo(trus_spots))[:-1,:]
    trus_spots_gt = trus_spots
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

    u = u_gt
    v = v_gt
    theta = theta_gt
    # calculate the solution
    if True:
        # trus_spots_noises = np.ones_like(trus_spots) * np.array([-1,-1,-2])[:,None]+\
        #                         np.random.rand(trus_spots.shape[0], trus_spots.shape[1])*np.array([2,2.0,4.0])[:,None]
        
        rotation_sigma = 0.05
        cam_spot_sigma = 0.05
        sigma_x = 0.005
        sigma_y = 0.005
        sigma_z = 0.01
        u_sigma = 1e-3 # unit: mm
        v_sigma = 1e-3 # unit: mm
        theta_noise = np.clip(rotation_sigma * np.random.randn(trus_spots.shape[1]),-2,2)+ theta_gt
        u_noise = np.clip(u_sigma * np.random.randn(trus_spots.shape[1]),-2,2)+ u_gt
        v_noise = np.clip(v_sigma * np.random.randn(trus_spots.shape[1]),-2,2) + v_gt
        trus_spots = np.concatenate([u_noise[None,:],        # x axis
                                    -1*v_noise*np.sin(theta_noise*np.pi/180.0)[None,:],  # y axis
                                        v_noise*np.cos(theta_noise*np.pi/180.0)[None,:]], axis=0) # z axis
        trus_spots_arc = np.concatenate([u_noise[None,:],        # x axis
                                    -1*v_noise*np.sin((theta_gt+random_theta)*np.pi/180.0)[None,:],  # y axis
                                        v_noise*np.cos((theta_gt+random_theta)*np.pi/180.0)[None,:]], axis=0) # z axis
        u = u_noise
        v = v_noise
        theta = theta_noise
        for i in range(cam_N.shape[1]):
            xx = np.clip(sigma_x * np.random.randn(),-0.01, 0.01)
            yy = np.clip(sigma_y * np.random.randn(),-0.01, 0.01)
            zz = np.clip(sigma_z * np.random.randn(),-0.06, 0.06)
            cam_N[:,i] = unit_vector(cam_N[:,i] + np.array([xx,yy,zz]))
        cam_laser_start_spots = cam_laser_start_spots + np.clip(cam_spot_sigma * np.random.randn(trus_spots.shape[1]),-3.5,3.5)

    # split the tracking set
    u = u[10:]
    v = v[10:]
    thetas_tracking = theta[10:]
    trus_spots = trus_spots[:,:10]
    trus_spots_tracking = trus_spots_arc[:,10:]
    trus_spots_arc = trus_spots_arc[:,:10] 
    line_num = int(line_num/2)
    cam_laser_start_spots_tracking = cam_laser_start_spots[:,10:]
    cam_laser_start_spots = cam_laser_start_spots[:,:10]
    cam_N_tracking = cam_N[:,10:]
    cam_N = cam_N[:,:10]


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
    tolerance = 0
    trus_spots_arc1 = trus_spots_arc
    error2s = [error1]
    errors_eval = [error1]
    while error2 < lst_error2: 
        if np.abs(error2 - lst_error2) < 6e-4:
            tolerance += 1
            if tolerance > 10:
                print('too small decline')
                break
        else:
            tolerance = 0
        count+=1
        theta = np.zeros(line_num)
        result = least_squares(fun, theta,\
                                bounds=([-6.0]*line_num, [6.0] * line_num ),\
                                    args=(trus_spots_arc, cam_spots_pred,F))
        thetas = result.x
        
        # print('estimated rotating angles: ', -1 * thetas)
        # print('real rotating angles: ', random_theta)
        
        # update trus_spots_arc
        trus_spots_arc = [rotation_matrix(theta*np.pi/180.0,[1,0,0],[0,0,0])[:3,:3] @ trus_spots_arc[:,i][:,None] \
                                        for i, theta in zip(range(line_num), thetas)]
        trus_spots_arc = np.concatenate(trus_spots_arc, axis=1)
        lst_error2 = error2
        lst_F2 = F2
        solver2 = Solver(geo_consist=False)
        F2,error2,lower2,upper2 = solver2.solve(trus_spots_arc, cam_laser_start_spots,cam_N, F0=F)
        # print('error2: ',error2 )
        error2s.append(error2)
        F = F2
        x_pred2, cam_spots_pred = solver2.output()
        trus_spots_pred2 = (np.linalg.inv(F2) @ homo(cam_spots_pred))[:3]
        error_eval = np.mean(np.linalg.norm(trus_spots_pred2 - trus_spots_gt[:,:line_num], axis = 0))
        errors_eval.append(error_eval)
        assert len(errors_eval) == len(error2s)
        print('validation error: ',error_eval )
        if error2 < 15e-2 or count > 30 :
            print("slight rotation")
            break
    F2=lst_F2
    # print(lst_error2,error1, error0)
    # print('x_pred1:', x_pred1[:,0])
    # print('x_pred2:', x_pred2[:,0])
    # print('x gt:', x)
    # print('translation error0:', np.linalg.norm(F0[:3,3] - Freg[:3,3]))
    # print('translation error1:', np.linalg.norm(F1[:3,3] - Freg[:3,3]))
    # print('translation error2:', np.linalg.norm(F2[:3,3] - Freg[:3,3]))
    # print(error2s)
    print(errors_eval)
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
    # print('rotation error0: ', angle0*180.0 /np.pi)
    # print('rotation error1: ', angle1*180.0 /np.pi )
    # print('rotation error2: ', angle2*180.0 /np.pi)
    trus_spots_pred1 = (np.linalg.inv(F1) @ homo(cam_spots_pred1))[:-1,:]
    trus_spots_pred2 = (np.linalg.inv(F2) @ homo(cam_spots_pred))[:-1,:]
    # for i in range(trus_spots_pred2.shape[1]):
    #     ax.scatter(trus_spots_pred2[0,i], trus_spots_pred2[1,i], trus_spots_pred2[2,i], marker='*',color=colors[i if i < 11 else 10])
    #     ax.scatter(trus_spots_pred1[0,i], trus_spots_pred1[1,i], trus_spots_pred1[2,i], marker='x',color=colors[i if i < 11 else 10])
    # plt.show()
    results = [np.linalg.norm(F0[:3,3] - Freg[:3,3]),np.linalg.norm(F1[:3,3] - Freg[:3,3]),np.linalg.norm(F2[:3,3] - Freg[:3,3]),\
                    angle0*180.0 /np.pi,angle1*180.0 /np.pi,angle2*180.0 /np.pi]
    

    print(trus_spots_gt[:,:10])
    print(trus_spots_pred2[:,:10])
    print('Freg: \n','['+';\n'.join([' '.join([str(c) for c in row ]) for row in np.linalg.inv(Freg)])+']')
    print('F2: \n','['+';\n'.join([' '.join([str(c) for c in row ])  for row in np.linalg.inv(F2)])+']')
    
    ##============================== Tracking ============================================###
    x = np.array([0,20])
    def tracking_fun(x,u,v,t,cam_laser_start_spots,cam_N,Freg):
        theta = x[:1] + t
        scale = x[1:]
        laser_spots_pred = cam_laser_start_spots + scale * cam_N
        trus_spot = np.array([u,-(v) * np.sin(theta*np.pi/180.0), (v) * np.cos(theta*np.pi/180.0)])[:,None]  # unit: mm
        diff = (Freg @ homo(trus_spot))[:3] - laser_spots_pred
        diff = np.array(diff)
        error = np.linalg.norm(diff) #+ 0.1 *np.sum( theta**2)

        return error
    tres0     = []
    rot_errs0 = []
    tres1 = []
    rot_errs1 = []
    tres2 = []
    rot_errs2 = []


    for idx in range(cam_laser_start_spots_tracking.shape[1]):
        uu = u[idx]
        vv = v[idx]
        t = thetas_tracking[idx]
        cam_laser_start_spot_tracking = cam_laser_start_spots_tracking[:,idx]
        cam_N_i = cam_N_tracking[:,idx]
        result = least_squares(tracking_fun, x,\
                                    bounds=([-17,0], [17,300] ),\
                                        args=(uu,vv,t,cam_laser_start_spot_tracking[:,None],cam_N_i[:,None],F1))
        rot_err = np.abs(result.x[0]+t-theta_gt[line_num+idx])
        t_pred = result.x[0] + t
        trus_spot_pred = np.array([uu,-(vv) * np.sin(t_pred*np.pi/180.0), (vv) * np.cos(t_pred*np.pi/180.0)])[:,None]
        
        tre = np.linalg.norm(trus_spots_gt[:,line_num+idx] - trus_spot_pred[:,0]) 
        rot_errs1.append(rot_err)
        tres1.append(tre)


        result = least_squares(tracking_fun, x,\
                                    bounds=([-17,0], [17,300] ),\
                                        args=(uu,vv,t,cam_laser_start_spot_tracking[:,None],cam_N_i[:,None],F2))
        rot_err = np.abs(result.x[0]+t-theta_gt[line_num+idx])
        t_pred = result.x[0] + t
        trus_spot_pred = np.array([uu,-(vv) * np.sin(t_pred*np.pi/180.0), (vv) * np.cos(t_pred*np.pi/180.0)])[:,None]
        
        tre = np.linalg.norm(trus_spots_gt[:,line_num+idx] - trus_spot_pred[:,0]) 
        rot_errs2.append(rot_err)
        tres2.append(tre)

        result = least_squares(tracking_fun, x,\
                                    bounds=([-17,0], [17,300] ),\
                                        args=(uu,vv,t,cam_laser_start_spot_tracking[:,None],cam_N_i[:,None],F0))
        rot_err = np.abs(result.x[0]+t-theta_gt[line_num+idx])
        t_pred = result.x[0] + t
        trus_spot_pred = np.array([uu,-(vv) * np.sin(t_pred*np.pi/180.0), (vv) * np.cos(t_pred*np.pi/180.0)])[:,None]
        
        tre = np.linalg.norm(trus_spots_gt[:,line_num+idx] - trus_spot_pred[:,0]) 
        rot_errs0.append(rot_err)
        tres0.append(tre)
    # print([np.mean(rot_errs1), np.mean(tres1),np.mean(rot_errs2), np.mean(tres2)])
        
    
    return results+[np.mean(rot_errs0), np.mean(tres0),np.mean(rot_errs1), np.mean(tres1),\
            np.mean(rot_errs2), np.mean(tres2)]


if __name__ == '__main__':
    times = 1
    rotation_sigma = 0.7
    cam_spot_sigma = 5e-2
    cam_N_sigma = [0.1,0.1,0.1]
    u_sigma = 1e-3 # unit: mm
    v_sigma = 1e-3 # unit: mm
    max_iteration = 150
    # with open('./results/simulation_with_noise6.txt', 'w') as f:
    #     noise_input = 'noise setting: {0:} {1:} {2:} {3:} {4:}\n'.format(rotation_sigma,cam_spot_sigma,u_sigma,v_sigma,max_iteration)
    #     f.write(noise_input)
    # f.close()

    for time in tqdm.tqdm(range(times)):
        result = main()
        # results.append(result)
        print(result)
        # with open('./results/simulation_with_slight_noise_40.txt', 'a') as f:
        
        #     result_input = ' '.join([str(val) for val in result])
        #     result_input += '\n'
        #     f.write(result_input)
    # f.close()
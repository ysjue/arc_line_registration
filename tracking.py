import os
import numpy as np
import scipy.stats as st
import yaml
from utils.transformations import rotation_matrix, quaternion_matrix, unit_vector
from Solver.point_line_registration import homo
from scipy.optimize import least_squares

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

def tracking(idx):
    with open('./data/luck_sample.yaml') as stream:
        try:
            data = yaml.safe_load((stream))
        except yaml.YAMLERROIR as exc:
            print(exc)
    data_list = data['Samples']
    # print(data_list)

    data_list = [d for i,d in zip(range(len(data_list)),data_list) if i >= 15]
    trus_spots = []
    keys = ['TRUS1', 'TRUS2', 'TRUS3']# 'TRUS4']
    thetas,us, vs = [],[],[]
    for d in data_list:
        key = random.sample(keys,1)[0] # 'TRUS1'
        key = 'TRUS2' 
        theta = d[key]['angle']
        u = d[key]['u']
        v = d[key]['v']
        y = -1*(0.01 + v ) * math.sin(theta/180.0 * math.pi)
        z = (0.01 + v ) * math.cos(theta/180.0 * math.pi)
        trus_spots.append([u, y, z])
        us.append(u)
        vs.append(v)
        thetas.append(theta)
    trus_spots = np.array(trus_spots).T * 1000 # convert to mm

    trus_spots_gt = []
    us_gt = []
    vs_gt = []
    thetas_gt = []
    for d in data_list:
        key = random.sample(keys,1)[0] # 'TRUS1'
        key = 'TRUS1'
        theta = d[key]['angle']
        u = d[key]['u']
        v = d[key]['v']
        y = -1*(0.01 + v ) * math.sin(theta/180.0 * math.pi)
        z = (0.01 + v ) * math.cos(theta/180.0 * math.pi)
        trus_spots_gt.append([u, y, z])
        us_gt.append(u)
        vs_gt.append(v)
        thetas_gt.append(theta)
    trus_spots_gt = np.array(trus_spots_gt).T * 1000 # convert to mm


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

    direc_vec = unit_vector( np.array([0.09555974, -0.05413993, -0.9939503]))

    cam_N = [unit_vector(cam2marker_transforms[i][:3,:3] @ direc_vec[:,None]) for i in range(len(cam2marker_transforms))]
    cam_laser_start_spots = [cam2marker_transforms[i] @ np.array([7.08929, 58.31985, -2.49508,1])[:,None] \
                                for i in range(len(cam2marker_transforms))]
    # fitted results:   [ 0.05312492  0.01149978 -0.02062431] [-0.35008202 -0.05933228 -0.93483809]
    cam_N = np.concatenate(cam_N,axis=1)
    cam_laser_start_spots = np.concatenate(cam_laser_start_spots,axis = 1)[:3]


    t  = thetas[idx]
    u  = us[idx]
    v = vs[idx]

    t_gt  = thetas_gt[idx]
    u_gt  = us_gt[idx]
    v_gt = vs_gt[idx]

    trus_spot_gt = np.array([u_gt, -1*(v_gt+0.01)*np.sin(t_gt*np.pi/180), (v_gt+0.01)*np.cos(t_gt*np.pi/180)])*1000
    cam_laser_start_spot = cam_laser_start_spots[:,idx] 
    cam_N = cam_N[:,idx]
    def fun(x,u,v,t,cam_laser_start_spots,cam_N,Freg):
        theta = x[:1] + t
        scale = x[1:]
        laser_spots_pred = cam_laser_start_spots + scale * cam_N
        trus_spot = np.array([u,-(v+0.01) * np.sin(theta*np.pi/180.0), (v+0.01) * np.cos(theta*np.pi/180.0)])[:,None] * 1000 # unit: mm
        diff = (Freg @ homo(trus_spot))[:3] - laser_spots_pred
        diff = np.array(diff)
        error = np.linalg.norm(diff) #+ 0.1 *np.sum( theta**2)

        return error


    # point-line 10 point, TRE: 1.80
    Freg = np.array([[-2.36531074e-03,  9.26348529e-01 , 3.76660071e-01, -3.32794553e+01],
    [ 9.77805890e-01 , 8.10530286e-02, -1.93199504e-01,  6.40626874e+00],
    [-2.09499516e-01 , 3.67843459e-01, -9.05980763e-01,  2.03332266e+02],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00 , 1.00000000e+00]])
    # # point-line 15 point, TRE: 3.69
    # Freg = np.array([[ 6.48477093e-02 , 8.12709637e-01, -5.79049066e-01,  1.71682718e+01],
    # [ 9.35328213e-01, -2.51738450e-01 ,-2.48573706e-01,  1.03464773e+01],
    # [-3.47787161e-01 ,-5.25481492e-01, -7.76474914e-01  ,1.97473407e+02],
    # [ 0.00000000e+00,  0.00000000e+00 , 0.00000000e+00 , 1.00000000e+00]])

    # # arc-line 10 point, accurate points,  TRE: 1.47
    # Freg = np.array([[-9.48398446e-03 , 9.71089671e-01 , 2.38526530e-01, -2.62801247e+01],
    # [ 9.66789899e-01,  6.98290498e-02, -2.45847908e-01,  9.01649989e+00],
    # [-2.55396445e-01,  2.28273423e-01, -9.39501943e-01,  2.04977649e+02],
    # [ 0.00000000e+00 , 0.00000000e+00 , 0.00000000e+00 , 1.00000000e+00]])

#     # arc-line 10 point, no accurate point
#     Freg = np.array([[ 2.51707297e-03,  9.99412071e-01 ,-3.41932354e-02, -1.37519368e+01],
#  [ 9.01838121e-01, -1.70425255e-02 ,-4.31737833e-01 , 1.78735474e+01],
#  [-4.32066741e-01, -2.97500476e-02, -9.01350801e-01,  1.99987739e+02],
#  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00 , 1.00000000e+00]])

    Freg = np.array([[-2.66271963e-02,  9.97110210e-01,  7.11492862e-02, -1.77898940e+01],
    [ 9.41961687e-01, 4.88549737e-02 ,-3.32146612e-01,  1.31517626e+01],
    [-3.34662774e-01,  5.81757686e-02, -9.40540487e-01 , 2.03995086e+02],
    [ 0.00000000e+00 , 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    # Freg = np.array([[ 6.97640532e-04,  9.66712168e-01,  2.55865391e-01, -2.73570441e+01],
    # [ 9.73007788e-01,  5.83901840e-02, -2.23263141e-01,  7.76182329e+00],
    # [-2.30771222e-01,  2.49114776e-01, -9.40577733e-01,  2.03888598e+02],
    # [ 0.00000000e+00 , 0.00000000e+00 , 0.00000000e+00 , 1.00000000e+00]])
#     Freg = np.array([[-1.03705490e-02,  7.59880262e-01,  6.49980338e-01, -4.64462883e+01],
#  [ 9.83816888e-01,  1.24032359e-01, -1.29307019e-01,  3.35103835e+00],
#  [-1.78876446e-01,  6.38120648e-01, -7.48869318e-01,  1.96195325e+02],
#  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    
    #point=4
#     Freg = np.array([[-1.40737120e-01,  7.86125347e-01,  6.01830543e-01, -4.58688190e+01],
#  [ 9.86485339e-01,  1.62863093e-01,  1.79523954e-02, -7.51135731e+00],
#  [-8.39031504e-02 , 5.96223575e-01, -7.98422012e-01,  1.86088891e+02],
#  [ 0.00000000e+00,  0.00000000e+00 , 0.00000000e+00,  1.00000000e+00]])
#     Freg = np.array([[-2.15748957e-01,  8.46250974e-01,  4.87146463e-01, -3.90944269e+01],
#  [ 9.76407306e-01,  1.82370548e-01,  1.15627658e-01, -1.25515912e+01],
#  [ 9.00885067e-03,  5.00599912e-01, -8.65631890e-01 , 1.93874028e+02],
#  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00 , 1.00000000e+00]])
#     Freg= np.array([[-2.36180300e-01 , 8.57304894e-01 , 4.57435443e-01, -3.75715664e+01],
#  [ 9.71651179e-01 , 2.03214629e-01,  1.20821357e-01, -1.27606358e+01],
#  [ 1.06231671e-02 , 4.73003312e-01 ,-8.80996604e-01 , 1.95185530e+02],
#  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00 , 1.00000000e+00]])

    #point=6
    Freg= np.array([[-1.13197928e-01 , 1.90143266e-01 ,-9.75208576e-01,  3.96662039e+01],
 [ 9.71520333e-01, -1.84461333e-01 ,-1.48735535e-01 , 4.69187714e+00],
 [-2.08169334e-01, -9.64271515e-01 ,-1.63847409e-01 , 1.66151949e+02],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00 , 1.00000000e+00]])
#     Freg= np.array([[-1.18150249e-01,  1.70077445e-01, -9.78322126e-01 , 3.96454463e+01],
#  [ 9.75907411e-01 ,-1.62102412e-01, -1.46039493e-01,  4.05583801e+00],
#  [-1.83426400e-01, -9.72006415e-01, -1.46827398e-01 , 1.63683432e+02],
#  [ 0.00000000e+00 , 0.00000000e+00 , 0.00000000e+00 , 1.00000000e+00]])
    Freg = np.array([[ 2.19889727e-01 , 7.48899591e-01, -6.25138313e-01,  2.09853057e+01],
 [ 8.73054044e-01 , 1.34827598e-01 , 4.68613012e-01 ,-2.94122164e+01],
 [ 4.35229990e-01, -6.48822720e-01, -6.24182613e-01 , 1.93145691e+02],
 [ 0.00000000e+00 , 0.00000000e+00  ,0.00000000e+00,  1.00000000e+00]])
    


    Freg = np.array( [[-0.010335818098238914 ,0.48586662238636646 ,-0.8739718508711274 ,33.49275036054526],
[0.9747329848019918 ,-0.190138199482201 ,-0.1172308553097976 ,3.696467841454751],     
[-0.2231339938315991 ,-0.8531008676285182 ,-0.4716249892109504 ,182.6834286746751],   
[0.0 ,0.0 ,0.0 ,1.0]])

    x = np.array([0,20])
    result = least_squares(fun, x,\
                                bounds=([-6,0], [6,300] ),\
                                    args=(u,v,t,cam_laser_start_spot[:,None],cam_N[:,None],Freg))
    rot_err = np.abs(result.x[0]+t-thetas_gt[idx])
    t_pred = result.x[0] + t
    trus_spot_pred = np.array([u,-(v+0.01) * np.sin(t_pred*np.pi/180.0), (v+0.01) * np.cos(t_pred*np.pi/180.0)])[:,None] * 1000
    tre = np.linalg.norm(trus_spot_gt - trus_spot_pred[:,0])
    print(tre,rot_err)

    return rot_err, tre


if __name__=='__main__':
    tres = []
    rot_errs = []
    for idx in range(5):
        rot_err,tre = tracking(idx)
        tres.append(tre)
        rot_errs.append(rot_err)
    a = [np.mean(tres), np.mean(rot_errs),]
    print('['+','.join([str(val) for val in a])+']')


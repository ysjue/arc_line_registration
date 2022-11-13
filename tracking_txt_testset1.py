from email.mime import multipart
import os
import numpy as np
import scipy.stats as st
import yaml
from utils.transformations import rotation_matrix, quaternion_matrix, unit_vector
from Solver.point_line_registration import homo
from scipy.optimize import least_squares


def main(idx = 0):
    file_path = './data/testset1'
    trus_samples_files = [f for f in os.listdir(file_path) if 'sample' in f and 'cam' not in f]
    trus_samples_txt = []

    # regorder the input file
    for i in range(len(trus_samples_files)):
        txt = [f for f in trus_samples_files if '-'+str(i+1)+'.txt' in f]
        assert len(txt) == 1
        trus_samples_txt.append(txt[0])
        
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


    for ii, sample_txt in enumerate(trus_samples_txt):
        sample_txt = os.path.join(file_path, sample_txt)
        sample = []
        with open(sample_txt,'r') as f:
            lines = f.readlines()
            # lines = [l.split('\n')[0] for l in lines]

        for line in lines:
            line = [float(l) for l in line.split(' ') if l != '' ]
            sample.append(line)
            
        sample = np.array(sample)
        samples.append(sample)
    
    # print(samples)

    direc_vec = unit_vector(np.array([-0.32871691, -0.10795516, -0.93823]))

    cam_Ns = [unit_vector(cam2marker_transforms[i][:3,:3] @ direc_vec[:,None]) for i in range(len(cam2marker_transforms))]
    cam_laser_start_spots = [cam2marker_transforms[i] @ np.array([ 49.9973,  18.979, -19.69427,1])[:,None] \
                                for i in range(len(cam2marker_transforms))]
    # fitted results:  [ -0.32871691, -0.10795516, -0.93823] [49.9973,  18.979, -19.69427,1]
    cam_Ns = np.concatenate(cam_Ns,axis=1)
    cam_laser_start_spots = np.concatenate(cam_laser_start_spots,axis = 1)[:3]


    gt_sample = [[sample[0,:][np.argmax(sample[-1,:])], sample[1,:][np.argmax(sample[-1,:])],\
                        sample[2,:][np.argmax(sample[-1,:])]] for sample in samples]
    gt_sample = np.array(gt_sample)
    
    selected_samples = []
    for sample in samples:
        weight = 1 * (np.random.rand() > 0.5)
        deviation = int((-5 + np.random.rand()*4)*weight + (1-weight)*(1+ np.random.rand()*4))
        index = np.argmax(sample[-1,:])+deviation
        selected_samples.append(sample[:3,index])
    sample = np.array(selected_samples)

    
    gt_theta  = gt_sample[idx,0]
    gt_u  = gt_sample[idx,1]
    gt_v = gt_sample[idx,2]

    t = sample[idx,0]
    u = sample[idx,1]
    v = sample[idx,2]
    trus_spot_gt = np.array([gt_u, -1*(gt_v+0.01)*np.sin(gt_theta*np.pi/180), (gt_v+0.01)*np.cos(gt_theta*np.pi/180)])*1000
    cam_laser_start_spot = cam_laser_start_spots[:,idx] 
    cam_N = cam_Ns[:,idx]
    def fun(x, u, v, sampled_theta, cam_laser_start_spots, cam_N, Freg):
        theta = x[:1]
        scale = x[1:]
        laser_spots_pred = cam_laser_start_spots + scale * cam_N
        trus_spot = np.array([u,-(0.01+v) * np.sin((sampled_theta+theta)*np.pi/180.0), (0.01+v) * np.cos((sampled_theta+theta)*np.pi/180.0)])[:,None] * 1000 # unit: mm
        diff = (Freg @ homo(trus_spot))[:3] - laser_spots_pred
        diff = np.array(diff)
        error = np.linalg.norm(diff) #+ 0.1 *np.sum( theta**2)
        # print(error)
        return error


    Freg1 = np.array([[-5.11389303e-01,  8.34014301e-01 , 2.07125871e-01, -1.03291442e+01],
    [ 8.58427904e-01 , 5.06938286e-01 , 7.81991510e-02 ,-3.77768171e+01],
    [-3.97808236e-02 , 2.17792837e-01 ,-9.75183965e-01 , 2.22573185e+02],
    [ 0.00000000e+00,  0.00000000e+00 , 0.00000000e+00 , 1.00000000e+00]])
    Freg1 = np.array([[-5.07173219e-01,  8.55084447e-01,  1.07730748e-01, -4.43122490e+00],
    [ 8.59657118e-01,  4.93018959e-01,  1.33872871e-01,-4.12765597e+01],
    [ 6.13593091e-02,  1.60508239e-01, -9.85125444e-01,  2.21979905e+02],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

    Freg2 = np.array([[-4.95565366e-01,  8.42537551e-01 , 2.11057912e-01, -1.07281469e+01],
    [ 8.68272044e-01 , 4.86918007e-01 , 9.49447830e-02 ,-3.89869408e+01],
    [-2.27733530e-02,  2.30307031e-01, -9.72851503e-01,  2.21642210e+02],
    [ 0.00000000e+00,  0.00000000e+00 , 0.00000000e+00 , 1.00000000e+00]])

    Freg2 = np.array([[-5.07173219e-01,  8.55084447e-01,  1.07730748e-01, -4.43122490e+00],
    [ 8.59657118e-01,  4.93018959e-01,  1.33872871e-01,-4.12765597e+01],
    [ 6.13593091e-02,  1.60508239e-01, -9.85125444e-01,  2.21979905e+02],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
#     Freg2 = np.array([[-5.04858221e-01 , 8.46691849e-01 , 1.68021099e-01 ,-9.00139506e+00],
#  [ 8.54815052e-01,  5.17458579e-01,-3.90876802e-02 ,-3.49800620e+01],
#  [-1.20039179e-01,  1.23893228e-01, -9.85008154e-01,  2.25599780e+02],
#  [ 0.00000000e+00 , 0.00000000e+00 , 0.00000000e+00,  1.00000000e+00]])
    
#     Freg2 = np.array(   [[-0.5021406842652126 ,0.8518136699075632 ,0.14922534960345243 ,-7.028805666855411],
# [0.8643804649273582 ,0.48909278639560616 ,0.11676753892990142 ,-40.304931454331715],
# [0.026479143823542578 ,0.18762120896740536 ,-0.9818844824560478 ,222.0110812933708],
# [0.0 ,0.0 ,0.0 ,1.0]])
#     Freg1 = np.array(  [[-0.5162184423329482 ,0.8221263606412569 ,0.24005575796907477 ,-12.84381746919732],
# [0.8556655663288965 ,0.5071148222378367 ,0.10330051144953917 ,-39.60596610883313],
# [-0.036809759499266376 ,0.2587330752057988 ,-0.9652472415915816 ,222.4184186928947],
# [0.0 ,0.0 ,0.0 ,1.0]])
#     Freg2 = np.array(   [[-0.4999392266232083 ,0.8241315934220439 ,0.26621022971897884 ,-14.110559574606745],
# [0.8648415403372562 ,0.49136852079266174 ,0.1029858576752825 ,-39.485751224593365],
# [-0.04593342781102294 ,0.281716335163005 ,-0.958397634967791 ,222.03271790346312],
# [0.0 ,0.0 ,0.0 ,1.0]])

    x = np.array([0,20])
    result1 = least_squares(fun, x,\
                                bounds=([-5,0], [5,130] ),\
                                    args=(u,v,t,cam_laser_start_spot[:,None],cam_N[:,None],Freg1))
    x = np.array([0,20])
    result2 = least_squares(fun, x,\
                                bounds=([-5,0], [5,130] ),\
                                    args=(u,v,t,cam_laser_start_spot[:,None],cam_N[:,None],Freg2))
    t_pred1 = result1.x[0]+t
    t_pred2 = result2.x[0]+t
    # print(t,result1.x[0]+t,result2.x[0]+t,gt_theta)
    
    # u = gt_u
    # v = gt_v

    trus_spot_pred1 = np.array([u,-(v+0.01) * np.sin(t_pred1*np.pi/180.0), (v+0.01) * np.cos(t_pred1*np.pi/180.0)]) * 1000 
    trus_spot_pred2 = np.array([u,-(v+0.01) * np.sin(t_pred2*np.pi/180.0), (v+0.01) * np.cos(t_pred2*np.pi/180.0)]) * 1000 
    tre1 = np.linalg.norm(trus_spot_gt - trus_spot_pred1)
    tre2 = np.linalg.norm(trus_spot_gt - trus_spot_pred2)
    # print(tre1,tre2)
    return result1.x[0]+t, result2.x[0]+t, gt_theta, tre1, tre2


if __name__ == '__main__':
    
    multiple_times = 12
    tracking_err1 = []
    tracking_err2 = []
    reg_err1 = []
    reg_err2 = []
    import tqdm
    for time in tqdm.tqdm(range(multiple_times)):
        point_line_tracking_error = []
        arc_line_tracking_error = []
        costs1 = []
        costs2 = []
        for i in range(5):         
            t1, t2,gt_t,cost1,cost2 = main(idx = i)
            point_line_tracking_error.append(np.abs(gt_t - t1))
            arc_line_tracking_error.append(np.abs(gt_t - t2))
            costs1.append(cost1)
            costs2.append(cost2)
        # print(np.mean(point_line_tracking_error))
        # print(np.mean(arc_line_tracking_error))
        # print(np.mean(costs1))
        # print(np.mean(costs2))
        tracking_err1.append(np.mean(point_line_tracking_error))
        tracking_err2.append(np.mean(arc_line_tracking_error))
        reg_err1.append(np.mean(costs1))
        reg_err2.append(np.mean(costs2))

    lower, upper = st.t.interval(alpha=0.95, df=len(tracking_err1)-1, 
                            loc=np.mean(tracking_err1), 
                            scale=st.sem(tracking_err1)) 
    print(np.mean(tracking_err1),lower, upper,np.std(tracking_err1))

    lower, upper = st.t.interval(alpha=0.95, df=len(tracking_err2)-1, 
                            loc=np.mean(tracking_err2), 
                            scale=st.sem(tracking_err2)) 
    print(np.mean(tracking_err2), lower, upper,np.std(tracking_err2))

    lower, upper = st.t.interval(alpha=0.95, df=len(reg_err1)-1, 
                            loc=np.mean(reg_err1), 
                            scale=st.sem(reg_err1)) 
    print(np.mean(reg_err1),lower, upper,np.std(reg_err1))

    lower, upper = st.t.interval(alpha=0.95, df=len(reg_err2)-1, 
                            loc=np.mean(reg_err2), 
                            scale=st.sem(reg_err2)) 
    print(np.mean(reg_err2),lower, upper, np.std(reg_err2))
    Mean = [np.mean(tracking_err1),np.mean(tracking_err2),\
            np.mean(reg_err1),np.mean(reg_err2)]
    Std = [np.std(tracking_err1),np.std(tracking_err2),\
            np.std(reg_err1),np.std(reg_err2)]
    print('['+','.join([str(np.mean(tracking_err2)),str(np.std(tracking_err2))\
                ,str(np.mean(reg_err2)),str(np.std(reg_err2))])+']')
    print('Mean: \n' + '['+','.join([str(m) for m in Mean])+']')
    print('Std: \n' +'['+ ','.join([str(s) for s in Std])+']')
    


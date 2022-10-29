from email.mime import multipart
import os
import numpy as np
import scipy.stats as st
import yaml
from utils.transformations import rotation_matrix, quaternion_matrix, unit_vector
from Solver.point_line_registration import homo
from scipy.optimize import least_squares

designated_testing_pairs = ['1.txt','4.txt','8.txt','14.txt','11.txt']

def main(idx = 0):
    file_path = './data/testset2'
    trus_samples_files = [f for f in os.listdir(file_path) if 'sample' in f and 'cam' not in f]
    trus_samples_txt = []

    # regorder the input file
    for i in range(len(trus_samples_files)):
        txt = [f for f in trus_samples_files if 'sample'+str(i+1)+'.txt' in f]
        assert len(txt) == 1
        trus_samples_txt.append(txt[0])
    trus_samples_txt = [f for f in trus_samples_txt if np.any([d in f for d in designated_testing_pairs])]
    cam_txt = os.path.join(file_path, 'testset_cam.txt')
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
    cam2marker_transforms = [ transform  for i,transform in enumerate(cam2marker_transforms) \
                                if i+1 in [int(d.split('.txt')[0]) for d in designated_testing_pairs ]]

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

    direc_vec = unit_vector( np.array([ 0.03135577,  0.21365248, -0.97640639]))

    cam_Ns = [unit_vector(cam2marker_transforms[i][:3,:3] @ direc_vec[:,None]) for i in range(len(cam2marker_transforms))]
    cam_laser_start_spots = [cam2marker_transforms[i] @ np.array([37.64178, -14.98675, -2.07052,1])[:,None] \
                                for i in range(len(cam2marker_transforms))]
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
    trus_spot_gt = np.array([gt_u, -1*(gt_v+0.01)*np.sin(t*np.pi/180), (gt_v+0.01)*np.cos(t*np.pi/180)])*1000
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


    Freg1 = np.array( [[-5.41795455e-01,  7.96418572e-01,  2.68654318e-01, -1.12597650e+01],
 [ 7.73031580e-01,  5.97640512e-01, -2.12715758e-01, -2.18015419e+01],
 [-3.29969484e-01,  9.24298417e-02, -9.39455621e-01,  2.09685336e+02],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00 , 1.00000000e+00]])


    Freg2 = np.array([[-4.31002799e-01 , 7.74771111e-01,  4.62564928e-01, -2.16827573e+01],
 [ 8.70854722e-01,  4.91402675e-01 ,-1.16389585e-02 ,-3.27005024e+01],
 [-2.36323172e-01 , 3.97810428e-01 ,-8.86509008e-01,  2.07036847e+02],
 [ 0.00000000e+00,  0.00000000e+00 , 0.00000000e+00 , 1.00000000e+00]])



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
    print(result1.x[0]+t,result2.x[0]+t,gt_theta)
    # print(result1.cost,result2.cost)
    trus_spot_pred1 = np.array([u,-(v+0.01) * np.sin(t_pred1*np.pi/180.0), (v+0.01) * np.cos(t_pred1*np.pi/180.0)]) * 1000 
    trus_spot_pred2 = np.array([u,-(v+0.01) * np.sin(t_pred2*np.pi/180.0), (v+0.01) * np.cos(t_pred2*np.pi/180.0)]) * 1000 
    tre1 = np.linalg.norm(trus_spot_gt - trus_spot_pred1)
    tre2 = np.linalg.norm(trus_spot_gt - trus_spot_pred2)
    return result1.x[0]+t, result2.x[0]+t, gt_theta, tre1, tre2



if __name__ == '__main__':
    
    multiple_times = 100
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
    print(np.mean(tracking_err1),lower, upper)

    lower, upper = st.t.interval(alpha=0.95, df=len(tracking_err2)-1, 
                            loc=np.mean(tracking_err2), 
                            scale=st.sem(tracking_err2)) 
    print(np.mean(tracking_err2), lower, upper)

    lower, upper = st.t.interval(alpha=0.95, df=len(reg_err1)-1, 
                            loc=np.mean(reg_err1), 
                            scale=st.sem(reg_err1)) 
    print(np.mean(reg_err1),lower, upper)

    lower, upper = st.t.interval(alpha=0.95, df=len(reg_err2)-1, 
                            loc=np.mean(reg_err2), 
                            scale=st.sem(reg_err2)) 
    print(np.mean(reg_err2),lower, upper )
    


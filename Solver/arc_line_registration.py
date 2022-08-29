from utils.transformations import superimposition_matrix,translation_matrix,rotation_matrix,rotation_from_matrix
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import least_squares
import scipy.stats as st
import math

homo = lambda x: np.concatenate([x,np.ones((1,x.shape[1]), np.float32)], axis = 0) if x.shape[0] == 3 else False

class LeastSquare_Solver(object):
    def __init__(self, geo_consist= False, epsilon = 0.000001, max_iter = 1000):
        """
        Param
        ---------
        epsilon: termination condition
        geo_consist: bool. Enable the geometric consistency
        max_iter: max number of iteration value
         
        """
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.geo_consist = geo_consist 
        def fun(var,trus_radii,trus_x, laser_start_points,directions,Freg):
            num = len(var)
            x = var[:int(num/2)]
            theta = var[int(num/2):]
            assert len(x) == len(theta)
            y = trus_radii[-1,:][None,:] * np.sin(theta * np.pi/180.0)
            z = trus_radii[-1,:][None,:] * np.cos(theta * np.pi/180.0)
            trus_points = trus_x + np.concatenate([np.zeros_like(y),y,z],axis=0)
            laser_points = laser_start_points + x * directions
            
            error = np.linalg.norm((Freg @ homo(trus_points))[:3,:] - laser_points, axis = 0)
            assert len(error) == num/2
            return error
        self.fun = fun

    def eval(self, x, y):
        diff = x - y
       
        self.errs = [np.linalg.norm(diff[:,i]) for i in range(x.shape[1])]

        return np.mean(self.errs)

    def output(self):
        return self.x, self.laser_spots
    
    

    def solve(self, trus_radii:np.ndarray, \
                    trus_x: np.ndarray, \
                        laser_start_points:np.ndarray, \
                        directions:np.ndarray, \
                            F0:np.ndarray)->np.ndarray:
        """
        Param
        -----------
        trus_spots: 3xN ndarray.
            3d coordinates of laser spots w.r.t. the TRUS frame.
        laser_start_points: 3xN ndarray.
            3d coordinates of laser lines' start positions w.r.t. camera frame
        directions: 3xN ndarray.
            3d unit vector of directions of laser lines w.r.t. camera frame
        F0: 4x4 ndarray.
            Initial guess for the Transformation from TRUS to Camera.

        Return
        ---------
        Freg: 4x4 ndarray.
        """
        F = F0
        point_num = trus_radii.shape[1]
        laser_spots = np.zeros_like(trus_radii)
        trus_spots = trus_radii + trus_x
        N = np.zeros((directions.shape[1],3*directions.shape[1]))
        for i in range(N.shape[0]):
            N[i, i*3:(i+1)*3] = directions[:,i]
        N = N.T
        for ii in range(self.max_iter):
            homo_cam_spots = F @ homo(trus_spots)
            cam_spots = homo_cam_spots[:-1,:]
            error = self.eval(cam_spots, laser_spots)
            print(error)
            if error < self.epsilon:
                print('Converge. Registration successful')
                self.x = x
                self.laser_spots = laser_spots
                return F,error, 0,0
            
            var = np.ones(2 * point_num,np.float32)
            result = least_squares(self.fun, var,\
                        bounds=([0.0] * point_num + [-45.0]*point_num, [np.inf] * point_num + [45.0] * point_num ),\
                            args=(trus_radii,trus_x, laser_start_points,directions,F))
            var = result.x
            x, theta = var[:point_num], var[point_num:]
            # Update the laser spot w.r.t. the camera view
           
            v = laser_start_points.T.reshape(-1,1) + N @ x[:,None] 
            laser_spots = v.reshape(-1,3).T
            y = trus_radii[-1,:][None,:] * np.sin(theta * np.pi/180.0)
            z = trus_radii[-1,:][None,:] * np.cos(theta * np.pi/180.0)
            trus_spots = trus_x + np.concatenate([np.zeros_like(y),y,z],axis=0)
            # print('laser_spots:',laser_spots)
            # print('cam_spots:',cam_spots)
            # print('diff:',laser_spots - cam_spots )

            # using Horn's methods
 
            F = superimposition_matrix(trus_spots,laser_spots, usesvd=False)
               
        
        self.x = x
        self.laser_spots = laser_spots
        lower, upper = st.t.interval(alpha=0.95, df=len(self.errs)-1, 
                                 loc=np.mean(self.errs), 
                                 scale=st.sem(self.errs))  
        return F, error,lower, upper


from transformations import superimposition_matrix,translation_matrix,rotation_matrix,rotation_from_matrix
import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize

homo = lambda x: np.concatenate([x,np.ones((1,x.shape[1]), np.float32)], axis = 0) if x.shape[0] == 3 else False

class LeastSquare_Solver(object):
    def __init__(self, geo_consist= False, epsilon = 0.000001, max_iter = 2000):
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

    def eval(self, x, y):
        diff = x - y
        return np.mean([np.linalg.norm(diff[:,i]) for i in range(x.shape[1])])

    def output(self):
        return self.x, self.laser_spots

    def solve(self, trus_spots:np.ndarray, \
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
        point_num = trus_spots.shape[1]
        laser_spots = np.zeros_like(trus_spots)

        N = np.zeros((directions.shape[1],3*directions.shape[1]))
        for i in range(N.shape[0]):
            N[i, i*3:(i+1)*3] = directions[:,i]
        N = N.T
        for ii in range(self.max_iter):
            homo_cam_spots = F @ homo(trus_spots)
            cam_spots = homo_cam_spots[:-1,:]
            error = self.eval(cam_spots, laser_spots)
            if error < self.epsilon:
                print('Converge. Registration successful')
                self.x = x
                self.laser_spots = laser_spots
                return F,error
            
            # Construct the matrix A, B for solving V(lambda) = Freg @ P
            B = (cam_spots - laser_start_points).T.reshape(-1,1)

            # Construct the matrix AA, BB for solving v_i - v-j = R_reg @ (p_i - p_j)
            AA = np.zeros((int(point_num*(point_num - 1)/2)*3, point_num))
            row = 0
            BB =np.zeros((AA.shape[0],1))
            for i in range(point_num):
                for j in range(i+1,point_num):
                    AA[row*3:(row+1)*3,i] = directions[:,i]
                    AA[row*3:(row+1)*3,j] = -directions[:,j]
                    BB[row*3:(row+1)*3,0] =  cam_spots[:,i] - cam_spots[:,j] \
                        -laser_start_points[:,i] +laser_start_points[:,j]
                    row +=1
            assert row == point_num*(point_num - 1)/2
            
            # Concatenate the two matrix
            if self.geo_consist:
                A = np.concatenate((N,AA), axis = 0)
                B = np.concatenate((B,BB), axis = 0)
            else:
                A = N
        
            x,_,_,_= np.linalg.lstsq(A,B,rcond=None)
            
            # Update the laser spot w.r.t. the camera view 
            v = laser_start_points.T.reshape(-1,1) + N @ x 
            laser_spots = v.reshape(-1,3).T
            # print('laser_spots:',laser_spots)
            # print('cam_spots:',cam_spots)
            # print('diff:',laser_spots - cam_spots )

            # using Horn's methods
            F = superimposition_matrix(trus_spots,laser_spots, usesvd=False)
               
        self.x = x
        self.laser_spots = laser_spots
        return F, error


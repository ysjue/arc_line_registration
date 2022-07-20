import numpy as np
from scipy.optimize import fsolve
import math
from sympy import *
from transformations import unit_vector

class solver(object):

    def __init__(self,v,n,p):
        self.v = v
        self.n = n
        self.p = p

    def equations(self, x):
        equations = []
        x=x[:4]
        for i in range(len(x)):
            for j in range(i+1,len(x)):
                vec = (self.v[:,i]+x[i]*self.n[:,i] -self.v[:,j]-x[j]*self.n[:,j])
                equation = np.matmul(vec.T, vec) - np.linalg.norm(self.p[:,i] - self.p[:,j])**2
                equations.append(equation)
        return equations
    def __call__(self, x):
        x = fsolve(self.equations, x)
        return x

p = np.array([[1,6,19,12],[-4,10,-5,-9],[3,-2,1,7]], np.float32)
n = [unit_vector(np.array([0,1,-2])),\
    unit_vector(np.array([3,1,-1])),\
    unit_vector(np.array([-1,0,-1])),\
    unit_vector(np.array([3,-1,-2]))]

n = np.asarray(n, np.float32).T
x = np.array([-26.4,-22.5, -18,-6.7])
v = np.zeros_like(p)
for i in range(4):
    v[:,i] = p[:, i] + x[i] * n[:,i]
solver = solver(v,n,p)
result = solver(np.array([1,1,1,1,0,0]))
print(result)




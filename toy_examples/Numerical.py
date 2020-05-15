import autograd.numpy as np
from Kernels import *
from utils import *
from Functions import *

from Functions import Gibbs, GradGibbs


def one_d_feynman_kac(process):
    u_start = process["u_start"]
    x_low, x_high, t_low, t_highfor  = process["domain"] # closed interval for spatial
    num_x, num_t = process["num_x"], process["num_y"] # number of points to approximate
    sig = process["temperature"]

    delta_x, delta_t = (x_high - x_low)/float(num_x - 1), (t_high - t_low)/float(num_t - 1) # subtract one because we want to count endpoints


    # get potential_function and gradient
    U, grad_U = get_potential(process)

    # init u
    u = np.array([[u_start(x) for x in range(x_low, x_high)]]) # 1D is time and rest are space

    V = lambda x: np.linalg.norm(grad_U(x), axis=0)

    # create forward matrix
    forward_matrix = []
    for i in range(num_x):
        row = [1/(4 * delta_x**2), grad_U(np.array([[delta_x * i]]))[0][0]/(2 * delta_x), - sig**2/2. * 1/(2. * delta_x ** 2) - V(delta_x * i)[0] + 1,
         grad_U(delta_x * i)[0][0]/(2 * delta_x), sig**2 / 2. * 1/(4. * delta_x)]
        forward_matrix.append(np.concatenate([a, row, b]))



    for t in range(t_low, t_high+1):
        u_t = []




        for x in range(x_low, x_high+1):

            # compute the update step
            

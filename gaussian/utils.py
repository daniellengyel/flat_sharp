import numpy as np
import matplotlib.pyplot as plt
import torch

from dataloaders import *

import datetime
import socket, sys, os

from collections import defaultdict
from tensorflow.python.summary.summary_iterator import summary_iterator

def get_file_stamp():
    """Return time and hostname as string for saving files related to the current experiment"""
    host_name = socket.gethostname()
    mydate = datetime.datetime.now()
    return "{}_{}".format(mydate.strftime("%b%d_%H-%M-%S"), host_name)

# +++ algorithm +++

def sample_index_softmax(weights, positions, beta=1):
    probabilities = softmax(weights, beta)
    pos_filter = np.random.choice(list(range(len(positions))), len(positions), p=probabilities)
    return pos_filter


def softmax(weights, beta=-1):
    # normalize weights:
    weights /= np.sum(weights)

    sum_exp_weights = sum([np.exp(beta * w) for w in weights])
    probabilities = np.array([np.exp(beta * w) for w in weights]) / sum_exp_weights
    return probabilities


def weight_function_input_jacobian(grad):
    input_shape = grad.shape  # batch, filters, x_dim, y_dim
    grad = grad.reshape((input_shape[0], np.product(input_shape[1:]))).T

    return np.sum(np.linalg.norm(grad, axis=0))



def kish_effs(weights):
    """Assume weights are just a list of numbers"""
    N = len(weights)
    weights = np.array(weights)
    sum_weights = np.sum(weights)
    return 1/float(N) *  sum_weights**2 / weights.dot(weights)


def get_params_vec(net):
    list_params = list(net.parameters())
    param_vec = [list_params[i].view(list_params[i].nelement()).detach().numpy() for i in range(len(list_params))]
    return np.concatenate(param_vec, 0)

def get_grad_params_vec(net):
    list_params = list(net.parameters())
    param_vec = [list_params[i].grad.view(list_params[i].nelement()).detach().numpy() for i in range(len(list_params))]
    return np.concatenate(param_vec, 0)

def get_params_var(nets):
    nets_param_vecs = []
    for net_idx in range(len(nets)):
        param_vec = get_params_vec(nets[net_idx])
        nets_param_vecs.append(param_vec)
    if len(nets) == 1:
        return np.cov(np.array(nets_param_vecs)).reshape(1,1)
    return np.cov(np.array(nets_param_vecs))


# Viz 

def classification_regions_2d(v1, v2, center_image, alpha_min, alpha_max, beta_min, beta_max, N, net):
    """ 
    Returns the alpha (X) and beta (Y) range used in the basis of v1 and v2. Use meshgrid(X, Y) to get the corresponding 
    coordinates for the result. """
    
    
    alpha_range = np.linspace(alpha_min, alpha_max, N)
    beta_range = np.linspace(beta_min, beta_max, N)
    
    results = []
    
    mesh = np.array(np.meshgrid(alpha_range, beta_range))
    
    mesh_2d = mesh.reshape(2, N*N)
    
    max_batch = 250*250
    i = 1
    results = np.array([])
    
    net.eval()
    
    while N*N > max_batch*(i - 1): 
        lin_comb = torch.stack([v1, v2]).T.mm(torch.Tensor(mesh_2d[:, (i-1)*max_batch:i*max_batch])).T
        lin_comb = lin_comb.reshape(lin_comb.shape[0], 1, center_image.shape[1], center_image.shape[2])
        lin_comb += center_image
        curr_results = net(lin_comb)
        curr_results = torch.argmax(curr_results, 1).detach().numpy()
        
        results = np.concatenate([results, curr_results])
        i += 1
        
    mesh = np.array(np.meshgrid(alpha_range, beta_range))
    return alpha_range, beta_range, np.array(results).reshape(mesh.shape[1:]), v1, v2

# from: https://github.com/facebookresearch/jacobian_regularizer/blob/master/jacobian/jacobian.py
def _random_vector(C, B):
    '''
    creates a random vector of dimension C with a norm of C^(1/2)
    (as needed for the projection formula to work)
    '''
    if C == 1: 
        return torch.ones(B)
    v=torch.randn(B,C)
    arxilirary_zero=torch.zeros(B,C)
    vnorm=torch.norm(v, 2, 1,True)
    v=torch.addcdiv(arxilirary_zero, 1.0, v, vnorm)
    return v


def get_vec(img):
    return img.reshape(np.product(img.shape[1:]))

def get_orthogonal_basis(img1, img2, img3):
    """img1 is center. img_2 is first orthogonal vector. use graham schmidt to get second vector from img_3."""
    img1_to_img2 = get_vec(img2) - get_vec(img1)
    unit_img1_to_img2 = img1_to_img2 / np.linalg.norm(img1_to_img2)
    
    img1_to_img3 = get_vec(img3) - get_vec(img1)
    unit_img1_to_img3 = img1_to_img3 / np.linalg.norm(img1_to_img3)
    
    assert abs(unit_img1_to_img2.dot(unit_img1_to_img3)) != 1
    
    # get orthogonal vectors which span the above subspace. Use grahamschmidt
    v1 = unit_img1_to_img2
    v2 = unit_img1_to_img3 - v1.dot(unit_img1_to_img3) * v1
    v2 = v2 / np.linalg.norm(v2)
    
    return v1, v2

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# net predictions
def get_average_output(nets, inp):
    outs = [net(inp).detach().numpy() for net in nets]
    return np.mean(outs, axis=0)

def get_gaussian_data(means, covs, train_nums, test_nums):

    train_gaussian = GaussianMixture(means, covs, len(means) * [train_nums])
    test_gaussian = GaussianMixture(means, covs, len(means) * [test_nums])

    return train_gaussian, test_gaussian
    


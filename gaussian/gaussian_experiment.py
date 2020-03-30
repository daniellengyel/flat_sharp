import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from ray import tune

from utils import *
from gaussian_training import *
from gaussian_postprocessing import *

from dataloaders import GaussianMixture

import sys, os
import pickle

# # Set up folder in which to store all results
# folder_name = get_file_stamp()
# folder_path = os.path.join(os.getcwd(), "gaussian_experiments", folder_name)
# print(os.getcwd())
# print(folder_path)
# os.makedirs(folder_path)
#
# # Get Data
# gaussian_params = []
#
# cov_1 = np.array([[1, 1/2.], [1/2., 1]])
# cov_2= np.array([[1, 1/2.], [1/2., 1]])
#
# mean_1 = np.array([0,0])
# mean_2 = np.array([2, 0])
#
# means = [mean_1, mean_2]
# covs = [cov_1, cov_2]
# training_nums = 500
# test_nums = 100
#
# data = get_gaussian_data(means, covs, training_nums, test_nums)
#
# # Store the data in our folder as data.pkl
# with open(os.path.join(folder_path, "data.pkl"), "wb") as f:
#     pickle.dump(data, f)
#
#
# config = {}
#
# # setting hyperparameters
#
# # net
# inp_dim = 2
# out_dim = 2
# width = tune.grid_search([4, 32, 256])
# num_layers = tune.grid_search([1])
# config["net_name"] = "SimpleNet"
# config["net_params"] = [inp_dim, out_dim, width, num_layers]
#
# config["torch_random_seed"] = 1
#
# config["num_epochs"] = tune.grid_search([10, 50])
#
# config["batch_train_size"] = tune.grid_search([16])
# config["batch_test_size"] = tune.grid_search([100])
#
# config["ess_threshold"] = tune.grid_search([0.97, 0.95, 0.9])
#
# config["learning_rate"] = 0.001
# config["momentum"] = 0.9
#
# config["num_nets"] = 100  # would like to make it like other one, where we can define region to initialize
#
# config["softmax_beta"] = tune.grid_search([-75, -10, 0, 10, 75]) # e.g. negtive to prioritize low weights
#
# config["weight_type"] = "loss_gradient_weights"  # "input_output_forbenius", #
#
# tune.run(lambda config_inp: train(data, config_inp, folder_path), config=config)

# save analysis processsing
folder_path = os.path.join(os.getcwd(), "gaussian_experiments", "Mar25_21-01-10_kingfisher.doc.ic.ac.uk")

# runs = get_runs(folder_path)
#
# os.mkdir(os.path.join(folder_path, "analysis"))
#
# with open(os.path.join(folder_path, "analysis", "runs.pkl"), "wb") as f:
#     pickle.dump(runs, f)

print("Run Analysis Done.")

eig = get_eig(folder_path, -1, use_gpu=False)

print("Eig Analysis Done.")

with open(os.path.join(folder_path, "analysis", "eig.pkl"), "wb") as f:
    pickle.dump(eig, f)

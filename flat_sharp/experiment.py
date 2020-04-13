import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from ray import tune

from utils import *
from training import *
from postprocessing import *
from data_getters import get_data

import sys, os
import pickle


config = {}

# setting hyperparameters

# data specific
data_name = "MNIST"
inp_dim = 28 * 28
out_dim = 10

# net
width = tune.grid_search([256])
num_layers = tune.grid_search([1])
config["net_name"] = "SimpleNet"
config["net_params"] = [inp_dim, out_dim, width, num_layers]

config["torch_random_seed"] = 1

config["num_steps"] = tune.grid_search([1600]) # roughly 50 * 500 / 16

config["batch_train_size"] = tune.grid_search([16])
config["batch_test_size"] = tune.grid_search([100])

config["ess_threshold"] = tune.grid_search([0.9988])

config["learning_rate"] = 0.001
config["momentum"] = 0

config["num_nets"] = 100  # would like to make it like other one, where we can define region to initialize

config["softmax_beta"] = tune.grid_search([-50, 0, 50]) # e.g. negtive to prioritize low weights

config["weight_type"] = "loss_gradient_weights"  # "input_output_forbenius", #


# --- Set up folder in which to store all results ---
folder_name = get_file_stamp()
folder_path = os.path.join(os.getcwd(), "experiments", data_name, folder_name)
print(os.getcwd())
print(folder_path)
os.makedirs(folder_path)

# --- get data ---
train_data, test_data = get_data(data_name)
if data_name == "gaussian":
    # Store the data in our folder as data.pkl
    with open(os.path.join(folder_path, "data.pkl"), "wb") as f:
        pickle.dump((train_data, test_data), f)


tune.run(lambda config_inp: train(config_inp, folder_path, train_data, test_data), config=config)
# train(config, folder_path)


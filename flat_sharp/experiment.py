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
data_name = "FashionMNIST"

if data_name == "CIFAR10":
    num_channels = 3
    height = 32
    width = height
    out_dim = 10
    inp_dim = height * width * num_channels
elif (data_name == "MNIST") or (data_name == "FashionMNIST"):
    num_channels = 1
    height = 28
    width = height
    out_dim = 10
    inp_dim = height * width * num_channels
elif data_name == "gaussian":
    inp_dim = 2
    out_dim = 2

# net
config["net_name"] = "SimpleNet"

if config["net_name"] == "SimpleNet":
    width = tune.grid_search([512])
    num_layers = tune.grid_search([4])
    config["net_params"] = [inp_dim, out_dim, width, num_layers]
elif config["net_name"] == "LeNet":
    config["net_params"] = [height, width, num_channels, out_dim]

config["torch_random_seed"] = 1

config["num_steps"] = None # tune.grid_search([25000]) # roughly 50 * 500 / 16
config["mean_loss_threshold"] = 0.25

config["batch_train_size"] = tune.grid_search([128])
config["batch_test_size"] = tune.grid_search([100])

config["ess_threshold"] =  None # tune.grid_search([0.97])
config["sampling_tau"] = 100 # tune.grid_search([100, 500])

config["learning_rate"] = tune.grid_search(list(np.linspace(1e-2, 1, 20)))
config["momentum"] = 0

config["num_nets"] = 3  # would like to make it like other one, where we can define region to initialize

config["softmax_beta"] = tune.grid_search([-50, 0, 50]) # e.g. negtive to prioritize low weights

config["weight_type"] = "loss_gradient_weights"  # "input_output_forbenius", #


# --- Set up folder in which to store all results ---
folder_name = get_file_stamp()
cwd = os.environ["PATH_TO_FLAT_FOLDER"]
folder_path = os.path.join(cwd, "experiments", data_name, folder_name)
print(folder_path)
os.makedirs(folder_path)

# --- get data ---
train_data, test_data = get_data(data_name, vectorized=config["net_name"]=="SimpleNet")
if data_name == "gaussian":
    # Store the data in our folder as data.pkl
    with open(os.path.join(folder_path, "data.pkl"), "wb") as f:
        pickle.dump((train_data, test_data), f)


tune.run(lambda config_inp: train(config_inp, folder_path, train_data, test_data), config=config)
# train(config, folder_path)

# TODO have logging of what we want to achieve with the current experiment.
# add a new distance metric. Distance from permutations -- used to know how many networks are needed.
# could also tell us how far away the vallys are and how symmetric the space is.
# Repeat experiments in finding minima with SGD paper
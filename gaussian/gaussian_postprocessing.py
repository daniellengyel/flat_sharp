import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision

from utils import *
from gaussian_training import *

import yaml, os, sys

# get hessians...
import torch
from hessian_eigenthings import compute_hessian_eigenthings

import pickle
# +++ process experiment results +++

def temp_tb_to_dict(path_to_events_file):
    nets_dict_potential = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    nets_dict_acc = defaultdict(lambda: defaultdict(float))

    for e in summary_iterator(path_to_events_file):
        for v in e.summary.value:
            if ("Potential" in v.tag):
                net_num = int(v.tag.split("_")[1])  # Todo make this /
                name = v.tag.split("/")[1]

                if e.step not in nets_dict_potential[net_num]:
                    nets_dict_potential[net_num][e.step] = {}
                nets_dict_potential[net_num][e.step][name] = v.simple_value
            elif "Accuracy" in v.tag:
                net_num = int(v.tag.split("_")[1])  # Todo make this /

                nets_dict_acc[net_num][e.step] = v.simple_value

    return nets_dict_potential, nets_dict_acc

def get_models(model_folder_path, step):
    nets_dict = {}

    largest_step = -float("inf")
    for root, dirs, files in os.walk(model_folder_path):
        for name in files:
            name_split_underscore = name.split("_")
            largest_step = max(int(name_split_underscore[-1].split(".")[0]), largest_step)

            if name_split_underscore[-1].split(".")[0] == str(step):
                file_path = os.path.join(root, name)
                with open(file_path, "rb") as f:
                    net = torch.load(f)
                nets_dict[name_split_underscore[1]] = net
    if step == -1:
        return get_models(model_folder_path, largest_step)
    return nets_dict

def get_last_acc_potential(nets_dict_potential, nets_dict_acc, potential_type="total"):
    w = []
    a = []
    for nn in range(len(nets_dict_acc)):
        last_step = int(list(nets_dict_acc[nn].keys())[-2])
        a.append(nets_dict_acc[nn][last_step])
        w.append(nets_dict_potential[nn][last_step][potential_type])
    return w, a, last_step

def get_last(run_file_path):
    potential, acc = temp_tb_to_dict(run_file_path)
    return get_last_acc_potential(potential, acc)

# iterate through runs
def get_runs(experiment_folder):
    run_config_dir = {}
    run_acc_pot_dir = {}
    for root, dirs, files in os.walk("{}/runs".format(experiment_folder), topdown=False):
        if len(files) != 2:
            continue
        run_file_name, config_name = files
        curr_dir = os.path.basename(root)

        # get acc-potentials
        try:
            with open(os.path.join(root, config_name), "rb") as f:
                config = yaml.safe_load(f)
            acc_potentials_step = get_last(os.path.join(root, run_file_name))
            run_config_dir[curr_dir] = config
            run_acc_pot_dir[curr_dir] = acc_potentials_step
        except:
            print(root)

    return run_acc_pot_dir, run_config_dir

# get eigenvalues of final model.
def get_eig(experiment_folder, step, use_gpu=False):
    # init
    eigenvalue_dict = {}
    num_eigenthings = 5  # compute top 5 eigenvalues/eigenvectors
    loss = torch.nn.CrossEntropyLoss()
    with open(os.path.join(experiment_folder, "data.pkl"), "rb") as f:
        data = pickle.load(f)

    dataloader = DataLoader(data[0], batch_size=len(data[0]), shuffle=True)  # fix the batch size

    # iterate through models
    for root, dirs, files in os.walk("{}/models".format(experiment_folder), topdown=False):
        if len(files) == 0:
            continue

        try:
            curr_dir = os.path.basename(root)
            eigenvalue_dict[curr_dir] = {}
            models_dict = get_models(root, step)
            # get eigenvals
            for k, m in models_dict.items():
                eigenvals, eigenvecs = compute_hessian_eigenthings(m, dataloader,
                                                                   loss, num_eigenthings, use_gpu=use_gpu, mode="lanczos")
                eigenvalue_dict[curr_dir][k] = (eigenvals, eigenvecs)
        except:
            print(root)
            curr_dir = os.path.basename(root)
            eigenvalue_dict[curr_dir] = {}
            models_dict = get_models(root, step)
            # get eigenvals
            for k, m in models_dict.items():
                eigenvals, eigenvecs = compute_hessian_eigenthings(m, dataloader,
                                                                   loss, num_eigenthings, use_gpu=use_gpu, mode="lanczos")
                eigenvalue_dict[curr_dir][k] = (eigenvals, eigenvecs)

            print(curr_dir)
            with open(os.path.join(experiment_folder, "tmp.pkl"), "wb") as f:
                pickle.dump(eigenvalue_dict, f)
    return eigenvalue_dict


if __name__ == "__main__":
    # get test--accuracy and total weight.
    experiment_name = "Mar22_17-16-50_Daniels-MacBook-Pro-4.local"
    experiment_folder = "./gaussian_experiments/{}".format(experiment_name)



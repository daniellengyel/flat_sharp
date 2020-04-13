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

def get_models_old(model_folder_path, step):
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
        return get_models_old(model_folder_path, largest_step)
    return nets_dict


def get_models(model_folder_path, step):
    if step == -1:
        largest_step = -float("inf")
        for root, dirs, files in os.walk(model_folder_path):
            for sample_step_dir in dirs:
                name_split_underscore = sample_step_dir.split("_")
                if len(name_split_underscore) == 1:
                    continue
                largest_step = max(int(name_split_underscore[-1]), largest_step)

        step = largest_step

    resample_path = os.path.join(model_folder_path, "step_{}".format(step))

    nets_dict = {}
    for root, dirs, files in os.walk(os.path.join(resample_path, "models")):
        for net_file_name in files:
            net_idx = net_file_name.split("_")[1].split(".")[0]
            with open(os.path.join(root, net_file_name), "rb") as f:
                net = torch.load(f)
            nets_dict[net_idx] = net
    with open(os.path.join(resample_path, "sampled_idx.pkl"), "rb") as f:
        sampled_idx = pickle.load(f)

    return nets_dict, sampled_idx

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

def get_configs(experiment_folder):
    config_dir = {}
    for root, dirs, files in os.walk("{}/runs".format(experiment_folder), topdown=False):
        if len(files) != 2:
            continue
        curr_dir = os.path.basename(root)
        with open(os.path.join(root, "config.yml"), "rb") as f:
            config = yaml.safe_load(f)
        config_dir[curr_dir] = config

    return config_dir

# get eigenvalues of specific model folder.
def get_eig_old(experiment_folder, step, use_gpu=False):
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
            print(curr_dir)

            eigenvalue_dict[curr_dir] = {}
            models_dict = get_models(root, step)
            # get eigenvals
            for k, m in models_dict.items():
                eigenvals, eigenvecs = compute_hessian_eigenthings(m, dataloader,
                                                                   loss, num_eigenthings, use_gpu=use_gpu, mode="lanczos", max_steps=20)
                eigenvalue_dict[curr_dir][k] = (eigenvals, eigenvecs)
        except:
            # get eigenvals
            # in case it didn't converge
            try:
                for k, m in models_dict.items():
                    eigenvals, eigenvecs = compute_hessian_eigenthings(m, dataloader,
                                                                       loss, num_eigenthings, use_gpu=use_gpu, mode="lanczos", max_steps=40)
                    eigenvalue_dict[curr_dir][k] = (eigenvals, eigenvecs)
            except:
                print("error!")

        with open(os.path.join(experiment_folder, "tmp.pkl"), "wb") as f:
            pickle.dump(eigenvalue_dict, f)
    return eigenvalue_dict


# get eigenvalues of specific model folder.
def _get_eig(models, train_loader, test_loader, loss, num_eigenthings=5, use_gpu=False):
    eig_dict = {}
    acc_dict = {}
    # get eigenvals
    for k, m in models.items():
        try:
            eigenvals, eigenvecs = compute_hessian_eigenthings(m, train_loader,
                                                               loss, num_eigenthings, use_gpu=use_gpu, mode="lanczos",
                                                               max_steps=100)
            eig_dict[k] = (eigenvals, eigenvecs)
            acc_dict[k] = get_net_accuracy(m, test_loader)
        except:
            print("Error for net {}.".format(k))

    return eig_dict, acc_dict


# get eigenvalues of specific model folder.
def get_eig(experiment_folder, step, use_gpu=False):
    # init
    eigenvalue_dict = {}
    acc_dict = {}
    num_eigenthings = 5  # compute top 5 eigenvalues/eigenvectors
    loss = torch.nn.CrossEntropyLoss()
    with open(os.path.join(experiment_folder, "data.pkl"), "rb") as f:
        data = pickle.load(f)

    train_loader = DataLoader(data[0], batch_size=len(data[0]), shuffle=True)  # fix the batch size
    test_loader = DataLoader(data[1], batch_size=len(data[1]))
    # iterate through models
    for curr_dir in os.listdir("{}/resampling".format(experiment_folder)):
        root = os.path.join("{}/resampling".format(experiment_folder), curr_dir)
        print(curr_dir)
        models_dict, sampled_idx = get_models(root, step)
        eigenvalue_dict[curr_dir], acc_dict[curr_dir] = _get_eig(models_dict, train_loader, test_loader, loss, num_eigenthings, use_gpu)

        with open(os.path.join(experiment_folder, "eig_tmp.pkl"), "wb") as f:
            pickle.dump(eigenvalue_dict, f)
        with open(os.path.join(experiment_folder, "acc_tmp.pkl"), "wb") as f:
            pickle.dump(acc_dict, f)
    return eigenvalue_dict, acc_dict

def main(experiment_name):
    # # # save analysis processsing
    # folder_path = os.path.join(os.getcwd(), "gaussian_experiments", experiment_name)
    #
    # runs = get_runs(folder_path)
    #
    # # get runs
    # os.mkdir(os.path.join(folder_path, "analysis"))
    #
    # with open(os.path.join(folder_path, "analysis", "runs.pkl"), "wb") as f:
    #     pickle.dump(runs, f)
    #
    # print("Run Analysis Done.")
    #
    # # get eigenvalues
    # eig = get_eig(folder_path, -1, use_gpu=False)
    #
    # print("Eig Analysis Done.")
    #
    # with open(os.path.join(folder_path, "analysis", "eig.pkl"), "wb") as f:
    #     pickle.dump(eig, f)
    experiment_folder = "/Users/daniellengyel/flat_sharp/gaussian/gaussian_experiments/Apr03_17-38-00_Daniels-MacBook-Pro-4.local"
    # get_eig(experiment_folder, -1)

    # errors
    errs = ["1585931450.079084",
            "1585938556.9890182",
            "1585930974.450449",
            "1585928292.0912151",
            "1585928292.195146",
            "1585939435.9856498",
            "1585930892.8467379",
            "1585936404.000936",
            "1585928292.161804",
            "1585930890.959148",
            "1585933418.320897",
            "1585930956.739653",
            "1585928292.110432"]

    eigenvalue_dict = {}
    acc_dict = {}
    loss = torch.nn.CrossEntropyLoss()
    with open(os.path.join(experiment_folder, "data.pkl"), "rb") as f:
        data = pickle.load(f)

    train_loader = DataLoader(data[0], batch_size=len(data[0]))  # fix the batch size
    test_loader = DataLoader(data[1], batch_size=len(data[1]))

    for curr_dir in errs:
        root = os.path.join("{}/resampling".format(experiment_folder), curr_dir)
        print(curr_dir)
        models_dict, sampled_idx = get_models(root, -1)
        while True:
            try:
                eigenvalue_dict[curr_dir], acc_dict[curr_dir] = _get_eig(models_dict, train_loader, test_loader, loss)
                break
            except:
                print("Retry")
                continue

    with open(os.path.join(experiment_folder, "eig_tmp_errs.pkl"), "wb") as f:
        pickle.dump(eigenvalue_dict, f)
    with open(os.path.join(experiment_folder, "acc_tmp_errs.pkl"), "wb") as f:
        pickle.dump(acc_dict, f)

if __name__ == "__main__":
    main("")
    # import argparse
    #
    # parser = argparse.ArgumentParser(description='Postprocess experiment.')
    # parser.add_argument('exp_name', metavar='exp_name', type=str,
    #                     help='name of experiment')
    #
    # args = parser.parse_args()
    #
    # print(args)
    #
    # experiment_name = args.exp_name



import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision

from utils import *
from training import *
from data_getters import get_data

import yaml, os, sys, re

# get hessians...
import torch
from hessian_eigenthings import compute_hessian_eigenthings

import pickle
# +++ process experiment results +++
def tb_to_dict(path_to_events_file, names):
    tb_dict = {} # step, breakdown by /

    for e in summary_iterator(path_to_events_file):
        for v in e.summary.value:
            t_split = re.split('/+|_+', v.tag)
            if t_split[0] in names:
                tmp_dict = tb_dict
                t_split = [e.step] + t_split
                for i in range(len(t_split) - 1):
                    s = t_split[i]
                    if s not in tmp_dict:
                        tmp_dict[s] = {}
                        tmp_dict = tmp_dict[s]
                    else:
                        tmp_dict = tmp_dict[s]
                tmp_dict[t_split[-1]] = v.simple_value
    return tb_dict


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


# iterate through runs
def get_runs(experiment_folder, names):
    run_config_dir = {}
    run_dir = {}
    for root, dirs, files in os.walk("{}/runs".format(experiment_folder), topdown=False):
        if len(files) != 2:
            continue
        run_file_name = files[0] if ("tfevents" in files[0]) else files[1]
        curr_dir = os.path.basename(root)
        print(root)
        try:
            run_dir[curr_dir] = tb_to_dict(os.path.join(root, run_file_name), names)
            cache_data(experiment_folder, "runs", run_dir)
        except:
            print("Error for this run.")

    return run_dir, run_config_dir

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
def _get_eig(models, train_loader, test_loader, loss, num_eigenthings=5, use_gpu=False, full_dataset=True):
    eig_dict = {}
    acc_dict = {}
    # get eigenvals
    for k, m in models.items():
        print(k)
        try:
            eigenvals, eigenvecs = compute_hessian_eigenthings(m, train_loader,
                                                               loss, num_eigenthings, use_gpu=use_gpu, full_dataset=full_dataset , mode="lanczos",
                                                               max_steps=50)
            eig_dict[k] = (eigenvals, eigenvecs)
            acc_dict[k] = get_net_accuracy(m, test_loader)
        except:
            print("Error for net {}.".format(k))

    return eig_dict, acc_dict

def get_postprocessing_data(experiment_folder):
    data_type = experiment_folder.split("/")[-2]
    if experiment_folder == "MNIST":
        return get_data("MNIST")
    elif experiment_folder == "gaussian":
        with open(os.path.join(experiment_folder, "data.pkl"), "rb") as f:
            data = pickle.load(f)
        return data
    else:
        raise NotImplementedError("{} data type is not implemented.".format(data_type))


# get eigenvalues of specific model folder.
def get_eig(experiment_folder, step, num_eigenthings=5, use_gpu=False):
    # init
    eigenvalue_dict = {}
    acc_dict = {}
    loss = torch.nn.CrossEntropyLoss()

    # get data
    train_data, test_data = get_postprocessing_data(experiment_folder)
    train_loader = DataLoader(train_data, batch_size=1000, shuffle=True)  # fix the batch size
    test_loader = DataLoader(test_data, batch_size=len(test_data))

    # iterate through models
    for curr_dir in os.listdir("{}/resampling".format(experiment_folder)):
        root = os.path.join("{}/resampling".format(experiment_folder), curr_dir)
        print(curr_dir)
        models_dict, sampled_idx = get_models(root, step)
        eigenvalue_dict[curr_dir], acc_dict[curr_dir] = _get_eig(models_dict, train_loader, test_loader, loss, num_eigenthings, use_gpu, full_dataset=False)

        # cache data
        cache_data(experiment_folder, "eig", eigenvalue_dict)
        cache_data(experiment_folder, "acc", acc_dict)

    return eigenvalue_dict, acc_dict

def cache_data(experiment_folder, name, data):
    cache_folder = os.path.join(experiment_folder, "postprocessing", "cache")
    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)
    with open(os.path.join(cache_folder, "{}.pkl".format(name)), "wb") as f:
        pickle.dump(data, f)

# def log_msg(experiment_folder, msg):
#     cache_folder = os.path.join(experiment_folder, "postprocessing", "cache")
#     os.makedirs(cache_folder)



def get_config_to_id_map(configs):
    map_dict = {}

    for net_id in configs:
        conf = configs[net_id]
        tmp_dict = map_dict
        for k, v in conf.items():
            if isinstance(v, list):
                v = tuple(v)

            if k not in tmp_dict:
                tmp_dict[k] = {}
            if v not in tmp_dict[k]:
                tmp_dict[k][v] = {}
            prev_dict = tmp_dict
            tmp_dict = tmp_dict[k][v]
        prev_dict[k][v] = net_id
    return map_dict

def get_ids(config_to_id_map, config):
    if not isinstance(config_to_id_map, dict):
        return [config_to_id_map]
    p = list(config_to_id_map.keys())[0]

    ids = []
    for c in config_to_id_map[p]:
        if isinstance(config[p], list):
            config_compare = tuple(config[p])
        else:
            config_compare = config[p]
        if (config_compare is None) or (config_compare == c):
            ids += get_ids(config_to_id_map[p][c], config)
    return ids

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
    experiment_folder = "/Users/daniellengyel/flat_sharp/MNIST/experiments/{}".format("Apr07_15-53-47_Daniels-MacBook-Pro-4.local")
    get_eig(experiment_folder, -1)


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



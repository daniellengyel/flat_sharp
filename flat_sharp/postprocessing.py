import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torchvision

from sklearn.manifold import TSNE

from utils import *
from training import *
from data_getters import get_data

import yaml, os, sys, re

from data_getters import *

from pyhessian import hessian

# get hessians...
import torch
from hessian_eigenthings import compute_hessian_eigenthings

import pickle
# +++ process experiment results +++
def tb_to_dict(path_to_events_file, names):
    tb_dict = {} # step, breakdown by / and _

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

    return nets_dict

def get_all_models(experiment_folder, step):
    models_dict = {}
    # iterate through models
    for curr_dir in os.listdir("{}/resampling".format(experiment_folder)):
        if "DS_Store" in curr_dir:
            continue
        root = os.path.join("{}/resampling".format(experiment_folder), curr_dir)
        print(curr_dir)
        models_dict[curr_dir] = get_models(root, step)
    return models_dict

def _get_sample_idxs(model_folder_path):
    sample_idx_dir = {}
    for root, dirs, files in os.walk(model_folder_path, topdown=False):
        for sample_step_dir in dirs:
            name_split_underscore = sample_step_dir.split("_")
            if len(name_split_underscore) == 1:
                continue
            with open(os.path.join(model_folder_path, sample_step_dir, "sampled_idx.pkl"), "rb") as f:
                sample_idx_dir[name_split_underscore[-1]] = pickle.load(f)
    return sample_idx_dir

def get_sample_idxs(experiment_folder):
    # init
    sampled_idxs_dict = {}

    # iterate through models
    for curr_dir in os.listdir("{}/resampling".format(experiment_folder)):
        if "DS_Store" in curr_dir:
            continue
        root = os.path.join("{}/resampling".format(experiment_folder), curr_dir)
        print(curr_dir)
        sampled_idxs_dict[curr_dir] = _get_sample_idxs(root)

    return sampled_idxs_dict


# iterate through runs
def get_runs(experiment_folder, names):
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

    return run_dir

def get_configs(experiment_folder):
    config_dir = {}
    for root, dirs, files in os.walk("{}/runs".format(experiment_folder), topdown=False):
        if len(files) != 2:
            continue
        curr_dir = os.path.basename(root)
        with open(os.path.join(root, "config.yml"), "rb") as f:
            config = yaml.load(f)
        config_dir[curr_dir] = config

    return pd.DataFrame(config_dir).T

def get_postprocessing_data(experiment_folder, vectorized=True):
    data_type = experiment_folder.split("/")[-2]
    if data_type == "MNIST":
        return get_data("MNIST", vectorized)
    if data_type == "FashionMNIST":
        return get_data("FashionMNIST", vectorized)
    if data_type == "CIFAR10":
        return get_data("CIFAR10", vectorized)
    elif data_type == "gaussian":
        with open(os.path.join(experiment_folder, "data.pkl"), "rb") as f:
            data = pickle.load(f)
        return data
    else:
        raise NotImplementedError("{} data type is not implemented.".format(data_type))

# get eigenvalues of specific model folder.
def _get_eig(models, train_loader, test_loader, loss, num_eigenthings=5, use_gpu=False, full_dataset=True):
    eig_dict = {}
    # get eigenvals
    for k, m in models.items():
        print(k)
        eigenvals, eigenvecs = compute_hessian_eigenthings(m, train_loader,
                                                           loss, num_eigenthings, use_gpu=use_gpu,
                                                           full_dataset=full_dataset, mode="lanczos",
                                                           max_steps=100, tol=1e-2)
        try:
        #     eigenvals, eigenvecs = compute_hessian_eigenthings(m, train_loader,
        #                                                        loss, num_eigenthings, use_gpu=use_gpu, full_dataset=full_dataset , mode="lanczos",
        #                                                        max_steps=50)
            eig_dict[k] = (eigenvals, eigenvecs)
        except:
            print("Error for net {}.".format(k))

    return eig_dict

# get eigenvalues of specific model folder.
def get_eig(experiment_folder, step, num_eigenthings=5, use_gpu=False, FCN=False):
    # init
    eigenvalue_dict = {}
    loss = torch.nn.CrossEntropyLoss()

    # get data
    train_data, test_data = get_postprocessing_data(experiment_folder, FCN)
    train_loader = DataLoader(train_data, batch_size=5000, shuffle=True)  # fix the batch size
    test_loader = DataLoader(test_data, batch_size=len(test_data))

    # iterate through models
    for curr_dir in os.listdir("{}/resampling".format(experiment_folder)):
        if "DS_Store" in curr_dir:
            continue
        root = os.path.join("{}/resampling".format(experiment_folder), curr_dir)
        print(curr_dir)
        models_dict = get_models(root, step)
        eigenvalue_dict[curr_dir] = _get_eig(models_dict, train_loader, test_loader, loss, num_eigenthings, use_gpu, full_dataset=True)

        # cache data
        cache_data(experiment_folder, "eig", eigenvalue_dict)

    return eigenvalue_dict


def _get_trace(models, data_loader, loss, full_dataset=True):
    trace_dict = {}

    hessian_dataloader = []
    for i, (inputs, labels) in enumerate(data_loader):
        hessian_dataloader.append((inputs, labels))
        if not full_dataset:
            break

    # get trace
    for k, m in models.items():
        print(k)
        a = time.time()
        ts = []
        if full_dataset:
            trace = hessian(m, loss, dataloader=hessian_dataloader, cuda=False).trace()
        else:
            trace = hessian(m, loss, data=hessian_dataloader[0], cuda=False).trace()

        trace_dict[k] = trace

    return trace_dict


def get_trace(experiment_folder, step,  use_gpu=False, FCN=False):
    # init
    trace_dict = {}
    loss = torch.nn.CrossEntropyLoss()

    # get data
    train_data, test_data = get_postprocessing_data(experiment_folder, FCN)
    train_loader = DataLoader(train_data, batch_size=5000, shuffle=True)  # fix the batch size
    test_loader = DataLoader(test_data, batch_size=len(test_data))

    # iterate through models
    for curr_dir in os.listdir("{}/resampling".format(experiment_folder)):
        if "DS_Store" in curr_dir:
            continue
        root = os.path.join("{}/resampling".format(experiment_folder), curr_dir)
        print(curr_dir)
        models_dict = get_models(root, step)
        trace_dict[curr_dir] = _get_trace(models_dict, train_loader, loss, full_dataset=False)

        # cache data
        cache_data(experiment_folder, "trace", trace_dict)

    return trace_dict


def _get_acc(models, test_loader, loss):
    acc_dict = {}
    # get eigenvals
    for k, m in models.items():
        acc_dict[k] = get_net_accuracy(m, test_loader)
    return acc_dict

def get_acc(experiment_folder, step, FCN=False):
    # init
    acc_dict = {}
    loss = torch.nn.CrossEntropyLoss()

    # get data
    train_data, test_data = get_postprocessing_data(experiment_folder, FCN)
    train_loader = DataLoader(train_data, batch_size=5000, shuffle=True)  # fix the batch size
    test_loader = DataLoader(test_data, batch_size=len(test_data))

    # iterate through models
    for curr_dir in os.listdir("{}/resampling".format(experiment_folder)):
        if "DS_Store" in curr_dir:
            continue
        root = os.path.join("{}/resampling".format(experiment_folder), curr_dir)
        print(curr_dir)
        models_dict = get_models(root, step)
        acc_dict[curr_dir] = _get_acc(models_dict, test_loader, loss)

        # cache data
        cache_data(experiment_folder, "acc", acc_dict)

    return acc_dict

def _get_tsne(models):
    models_vecs = np.array([get_params_vec(m).detach().numpy() for k, m in sorted(models.items(), key=lambda item: int(item[0]))])
    X_embedded = TSNE(n_components=2).fit_transform(models_vecs)
    return X_embedded

def get_tsne(experiment_folder, step):
    # init
    tsne_dict = {}

    # iterate through models
    for curr_dir in os.listdir("{}/resampling".format(experiment_folder)):
        if "DS_Store" in curr_dir:
            continue

        root = os.path.join("{}/resampling".format(experiment_folder), curr_dir)
        print(curr_dir)
        models_dict = get_models(root, step)
        tsne_dict[curr_dir] = _get_tsne(models_dict)

        # cache data
        cache_data(experiment_folder, "tsne", tsne_dict)

    return tsne_dict



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

def get_all_model_steps(resampling_dir):
    step_dir = {}
    for root, dirs, files in os.walk(resampling_dir):
        for sample_step_dir in dirs:
            name_split_underscore = sample_step_dir.split("_")
            if len(name_split_underscore) == 1:
                continue
            step_dir[int(name_split_underscore[1])] = sample_step_dir
    return step_dir

# get all tsne embeddings
def get_tsne_dict(experiment_folder, curr_dir):
    tsne_dict = {}
    step_dir = get_all_model_steps(os.path.join(experiment_folder, "resampling", curr_dir))
    for step in sorted(step_dir):
        print(step)
        models = get_models(os.path.join(experiment_folder, "resampling", curr_dir), step)
        models_vecs = np.array([get_params_vec(m).detach().numpy() for m in models.values()])

        X_embedded = TSNE(n_components=2).fit_transform(models_vecs)

        tsne_dict[step] = X_embedded
    return tsne_dict

def _get_dirichlet_energy(nets, data, num_steps, step_size, var_noise, alpha=1, seed=1):
    """We use an OU process cetered at net.
    alpha is bias strength in OU."""
    # TODO add with noise that only comes from data_loader directions. i.e. same covariance as gradient RV.
    # TODO watch out with seed. we already are inputing nets and maybe other stuff?
    torch.manual_seed(seed)
    np.random.seed(seed)

    # init weights and save initial position.
    Xs_0 = [list(copy.deepcopy(n).parameters()) for n in nets]
    nets_weights = np.zeros(len(nets))
    criterion = torch.nn.CrossEntropyLoss()

    for i in range(num_steps):
        for idx_net in range(len(nets)):
            # get net and optimizer
            net = nets[idx_net]
            # Do OU step with EM discretization
            with torch.no_grad():
                for layer_idx, ps in enumerate(net.parameters()):
                    ps.data += torch.randn(ps.size()) * np.sqrt(var_noise*step_size)
                    ps.data += step_size*alpha*(Xs_0[idx_net][layer_idx].data - ps.data)
            nets_weights[idx_net] += unbiased_weight_estimate(net, data, criterion, num_samples=3, batch_size=500, max_steps=3) # max_steps and batch_size are from running some analysis. just checking
        print(i)
        # cache data
    return nets_weights

def different_cols(df):
    a = df.to_numpy() # df.values (pandas<0.24)
    a[:, 14] = 0
    return (a[0] != a[1:]).any(0)

def get_hp(cfs):
    filter_cols = different_cols(cfs)
    hp_names = cfs.columns[filter_cols]
    hp_dict = {hp: cfs[hp].unique() for hp in hp_names}
    return hp_dict

def get_dirichlet_energy(experiment_folder, model_step, num_steps=20, step_size=0.001, var_noise=0.5, alpha=1, seed=1, FCN=False):
    # init
    energy_dict = {}

    # get data
    train_data, test_data = get_postprocessing_data(experiment_folder, FCN)

    # iterate through models
    for curr_dir in os.listdir("{}/resampling".format(experiment_folder)):
        root = os.path.join("{}/resampling".format(experiment_folder), curr_dir)
        print(curr_dir)
        models_dict = get_models(root, model_step)
        nets = [v for k, v in sorted(models_dict.items(), key=lambda item: int(item[0]))]

        energy_dict[curr_dir] = _get_dirichlet_energy(nets, train_data, num_steps, step_size, var_noise, alpha=1, seed=1)

        # cache data
        cache_data(experiment_folder, "energy", energy_dict)

    return energy_dict


def get_stuff(experiment_folder):
    stuff = {}

    stuff_to_try = ["tsne", "runs", "trace"]

    for singular_stuff in stuff_to_try:
        try:
            with open(os.path.join(experiment_folder, "postprocessing/cache/{}.pkl".format(singular_stuff)), "rb") as f:
                stuff[singular_stuff] = pickle.load(f)
        except:
            print("Error: {} could not be found".format(singular_stuff))

    stuff["configs"] = get_configs(experiment_folder)

    return stuff

def main(experiment_name):
    # # # save analysis processsing
    # names = ["Loss", "Potential", "Accuracy", "Kish"]
    # run_data = get_runs(experiment_folder, names)

    root_experiment_folder = "/Users/daniellengyel/flat_sharp/flat_sharp/experiments/FashionMNIST/{}"
    exp =  "May14_23-57-49_Daniels-MacBook-Pro-4.local"
    experiment_folder = root_experiment_folder.format(exp)

    get_runs(experiment_folder , ["Loss", "Kish", "Potential", "Accuracy", "WeightVarTrace"]) # TODO does not find acc and var

    get_trace(experiment_folder, -1, 1, FCN=True)
    # get_dirichlet_energy(experiment_folder, -1, num_steps=20, step_size=0.001, var_noise=0.5, alpha=1, seed=1, FCN=True)
    get_tsne(experiment_folder, -1)


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



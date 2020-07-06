import pandas as pd
import pickle,os, copy
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import sys
sys.path.append("..")
from nets import Nets
from utils import *

import re

from sklearn.cluster import DBSCAN

from interpolation import *


from postprocessing import *
from analysis import *


def get_traces(net, data, device=None):
    criterion = torch.nn.CrossEntropyLoss()
    traces = []
    if device is not None:
        is_gpu = True
    else:
        is_gpu = False
    for i in range(len(data[0])):
        inputs, labels = data[0][i], data[1][i]
        inputs, labels = inputs.view(1, *inputs.shape), labels.view(1, *labels.shape)

        if device is not None:
            inputs, labels = inputs.to(device).type(torch.cuda.FloatTensor), labels.to(device).type(
                torch.cuda.LongTensor)

        traces.append(np.mean(hessian(net, criterion, data=(inputs, labels), cuda=is_gpu).trace()))
    return traces

if __name__ == "__main__":
    root_folder = os.environ["PATH_TO_FLAT_FOLDER"]
    data_name = "CIFAR10"
    exp = "SimpleNet_two_bs"
    experiment_folder = os.path.join(root_folder, "experiments", data_name, exp)

    # get all sorts of data

    # init torch
    is_gpu = True
    if is_gpu:
        torch.backends.cudnn.enabled = True
        device = torch.device("cuda:0")
    else:
        device = None
        # device = torch.device("cpu")

    exp_dict = {}

    exp_dict["stuff"] = get_stuff(experiment_folder)
    exp_dict["models"] = get_all_models(experiment_folder, -1)
    exp_dict["resampling_idxs"] = get_sample_idxs(experiment_folder)

    # get data
    train_data, test_data = get_postprocessing_data(experiment_folder, vectorized=True)
    train_loader = DataLoader(train_data, batch_size=10, shuffle=True)  # fix the batch size
    full_train_loader = DataLoader(train_data, batch_size=len(train_data), shuffle=True)  # fix the batch size

    test_loader = DataLoader(test_data, batch_size=1)

    criterion = torch.nn.CrossEntropyLoss()

    data = next(iter(train_loader))

    net = exp_dict["models"]["1594044887.584526"][str(0)]

    if device is not None:
        net = net.to(device)

    a = time.time()
    ts = get_traces(net, data, device=device)
    print(time.time() - a)

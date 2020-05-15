import numpy as np
import pandas as pd
import pickle,os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import sys
sys.path.append("..")
from nets import Nets
from utils import *

import re

from postprocessing import *


def get_end_stats(stuff):
    stats_dict = {}

    runs = stuff["runs"]
    if "trace" in stuff:
        trace = stuff["trace"] # assume the trace i get is from the end.
    else:
        trace = None
    configs = stuff["configs"]
    for exp_id in configs:
        num_nets = configs[exp_id]["num_nets"]
        num_steps = max(runs[exp_id], key=lambda x: int(x))


        stats_dict[str(exp_id)] = {}

        Loss_train_list = [runs[exp_id][num_steps - 1]["Loss"]["train"]["net"][str(nn)] for nn in range(num_nets)]
        Acc_test_list = [runs[exp_id][num_steps]["Accuracy"]["net"][str(nn)] for nn in range(num_nets)]

        stats_dict[dict_key]["Mean Train Loss"] = np.mean(Loss_train_list)
        stats_dict[dict_key]["Mean Test Acc"] = np.mean(Acc_test_list)

        try:
            Trace_list = [np.mean(trace[exp_id][str(nn)]) for nn in range(num_nets)]
            Trace_std_list = [np.std(trace[exp_id][str(nn)]) for nn in range(num_nets)]
            stats_dict[dict_key]["Mean Trace"] = np.mean(Trace_list)
            stats_dict[dict_key]["Mean Std Trace"] = np.mean(Trace_std_list)

            # print(dict_key)
            # print(trace[exp_id][str(0)])
            # print()

            stats_dict[dict_key]["Train Loss/Trace Correlation"] = get_correlation(Loss_train_list, Trace_list)
            stats_dict[dict_key]["Test Acc/Trace Correlation"] = get_correlation(Acc_test_list, Trace_list)
        except:
            print("Error: No trace for {}".format(exp_id))

    #         print("Mean Loss: {:.4f}".format(np.mean(Loss_list)))
    #         print("Mean Trace: {:.4f}".format(np.mean(Trace_list)))
    #         print("Mean Acc: {:.4f}".format(np.mean(Acc_list)))

    #         print("Loss/Trace Correlation: {:.4f}".format(get_correlation(Loss_list, Trace_list)))
    #         print("Acc/Trace Correlation: {:.4f}".format(get_correlation(Acc_list, Trace_list)))

    #         print("")

    stats_pd = pd.DataFrame(stats_dict).T

    cfs_hp = get_hp(configs)
    cfs_hp_df = configs[list(cfs_hp.keys())]
    stats_pd = pd.concat([stats_pd, cfs_hp_df], axis=1)

    return stats_pd


def what_it_do():
    plots = []
    plots_names = []

    stuff_stuff = [two_sampling]

    for stuff in stuff_stuff:
        for id_exp in stuff["configs"]:

            beta = stuff["configs"][id_exp]["softmax_beta"]
            plots_names.append(str(beta))


            X_embedded = stuff["tsne"][id_exp]

            plots.append(plt.scatter(X_embedded[:, 0], X_embedded[:, 1]))


    plt.legend(tuple(plots),
           plots_names,
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)



    plt.xlabel("loss")
    plt.ylabel("trace")

    plt.show()
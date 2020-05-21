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

import itertools


def get_end_stats(stuff):
    stats_dict = {}

    runs = stuff["runs"]
    if "trace" in stuff:
        trace = stuff["trace"] # assume the trace i get is from the end.
    else:
        trace = None

    if "acc" in stuff:
        accs = stuff["acc"]
    else:
        accs = None

    configs = stuff["configs"]
    for exp_id in configs.index:
        num_nets = configs.loc[exp_id]["num_nets"]
        try:
            num_steps = max(runs[exp_id], key=lambda x: int(x))
        except:
            continue

        stats_dict[str(exp_id)] = {}

        Loss_train_list = [runs[exp_id][num_steps - 1]["Loss"]["train"]["net"][str(nn)] for nn in range(num_nets)]

        stats_dict[str(exp_id)]["Mean Train Loss"] = np.mean(Loss_train_list)
        stats_dict[str(exp_id)]["Max Train Loss"] = np.max(Loss_train_list)
        stats_dict[str(exp_id)]["Min Train Loss"] = np.min(Loss_train_list)

        if accs is None:
            Acc_test_list = [runs[exp_id][num_steps]["Accuracy"]["net"][str(nn)] for nn in range(num_nets)]
        else:
            Acc_test_list = [accs[exp_id][str(nn)] for nn in range(num_nets)]

        stats_dict[str(exp_id)]["Mean Test Acc"] = np.mean(Acc_test_list)
        stats_dict[str(exp_id)]["Max Test Acc"] = np.max(Acc_test_list)
        stats_dict[str(exp_id)]["Min Test Acc"] = np.min(Acc_test_list)


        try:
            Trace_list = [np.mean(trace[exp_id][str(nn)]) for nn in range(num_nets)]
            Trace_std_list = [np.std(trace[exp_id][str(nn)]) for nn in range(num_nets)]
            stats_dict[str(exp_id)]["Mean Trace"] = np.mean(Trace_list)
            stats_dict[str(exp_id)]["Mean Std Trace"] = np.mean(Trace_std_list)
            stats_dict[str(exp_id)]["Max Trace"] = np.max(Trace_list)
            stats_dict[str(exp_id)]["Min Trace"] = np.min(Trace_list)


            # print(dict_key)
            # print(trace[exp_id][str(0)])
            # print()

            stats_dict[str(exp_id)]["Train Loss/Trace Correlation"] = get_correlation(Loss_train_list, Trace_list)
            stats_dict[str(exp_id)]["Test Acc/Trace Correlation"] = get_correlation(Acc_test_list, Trace_list)
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




def _plot(plots, plots_names, X_axis_name, Y_axis_name, X_axis_bounds, Y_axis_bounds):
    plt.legend(tuple(plots),
               plots_names,
               scatterpoints=1,
               loc='lower left',
               ncol=3,
               fontsize=8)

    plt.xlabel(X_axis_name)
    plt.ylabel(Y_axis_name)
    if X_axis_bounds is not None:
        plt.xlim(X_axis_bounds)
    if Y_axis_bounds is not None:
        plt.ylim(Y_axis_bounds)
    plt.show()


def plot_stats(stats_pd, X_axis_name, Y_axis_name, filter_by=None, seperate=False, X_axis_bounds=None,
               Y_axis_bounds=None):
    plots = []
    plots_names = []

    if ((filter_by is not None) and ("lr_bs_ratio" in filter_by)) or (X_axis_name == "lr_bs_ratio") or (Y_axis_name == "lr_bs_ratio"):
        stats_pd["lr_bs_ratio"] = stats_pd["learning_rate"] / stats_pd["batch_train_size"]

    if filter_by is None:
        x_values = stats_pd[X_axis_name].to_numpy()
        y_values = stats_pd[Y_axis_name].to_numpy()

        plots.append(plt.scatter(x_values, y_values))
        plots_names.append("Plot all")
        _plot(plots, plots_names, X_axis_name, Y_axis_name, X_axis_bounds, Y_axis_bounds)
    else:

        unique_filter_dict = {f: list(set(stats_pd[f])) for f in filter_by}
        unique_filter_keys = list(unique_filter_dict.keys())
        for comb in itertools.product(*unique_filter_dict.values()):

            filter_pd = stats_pd[(stats_pd[unique_filter_keys] == comb).to_numpy().all(1)]

            x_values = filter_pd[X_axis_name].to_numpy()
            y_values = filter_pd[Y_axis_name].to_numpy()

            plots.append(plt.scatter(x_values, y_values))
            plots_names.append(comb)
            if seperate:
                _plot(plots, plots_names, X_axis_name, Y_axis_name, X_axis_bounds, Y_axis_bounds)
                plots = []
                plots_names = []

        if not seperate:
            _plot(plots, plots_names, X_axis_name, Y_axis_name, X_axis_bounds, Y_axis_bounds)



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

def get_runs_plots_seperate(exp_dict, var_name="Kish", running_average_gamma=0.2):
    exp_runs = exp_dict["stuff"]["runs"]
    for i in exp_runs:

        plot_list = [0]

        print(i)

        for step in sorted(exp_runs[i], key=lambda x: int(x)):
            try:
                # going down the tree with node names given by var_name.split("/")
                curr_dict = exp_runs[i][step]
                var_name_split = var_name.split("/")
                for n in var_name_split:
                    curr_dict = curr_dict[n]

                if "net" in curr_dict:
                    num_nets = int(max(curr_dict["net"], key=lambda x: int(x))) + 1  # +1 bc zero indexed
                    to_append = np.array([curr_dict["net"][str(nn)] for nn in range(num_nets)])

                else:
                    to_append = curr_dict[""]
                to_append = plot_list[-1] * (1 - running_average_gamma) + running_average_gamma * to_append
                plot_list.append(to_append)
            except:
                print("No {} for step {}".format(var_name, step))
        plt.plot(plot_list[1:])
        plt.show()



def get_runs_plots(exp_dict, var_name="Kish", exp_ids= None, running_average_gamma=1, seperate=False):
    exp_runs = exp_dict["stuff"]["runs"]
    exp_plots = []
    exp_plots_names = []
    for i in exp_runs:

        plot_list = None

        if (exp_ids is not None) and (i not in exp_ids):
            continue

        print(i)

        for step in sorted(exp_runs[i], key=lambda x: int(x)):
            try:
                # going down the tree with node names given by var_name.split("/")
                curr_dict = exp_runs[i][step]
                var_name_split = var_name.split("/")
                for n in var_name_split:
                    curr_dict = curr_dict[n]
                if "net" in curr_dict:
                    num_nets = int(max(curr_dict["net"], key=lambda x: int(x))) + 1  # +1 bc zero indexed
                    to_append = np.array([curr_dict["net"][str(nn)] for nn in range(num_nets)])
                    # to_append = np.mean(to_append)

                else:
                    to_append = curr_dict[""]
                if plot_list is None:
                    plot_list = [to_append]
                else:
                    to_append = plot_list[-1] * (1 - running_average_gamma) + running_average_gamma * to_append
                    plot_list.append(to_append)
            except:
                print("No {} for step {}".format(var_name, step))
        if seperate:
            print(i)
            plt.plot(plot_list)
            plt.show()
        else:
            exp_plots.append( plt.scatter(list(range(len(plot_list))), plot_list))
            exp_plots_names.append(i)
    if not seperate:
        plt.legend(tuple(exp_plots),
                   exp_plots_names,
                   # scatterpoints=1,
                   loc='lower left',
                   ncol=3,
                   fontsize=8)

        plt.show()


def get_stat_step(exp_dict, var_name, step, exp_ids= None):

    if var_name == "trace":
        res_dict = {}
        if exp_ids is None:
            exp_ids = list(exp_dict["stuff"]["trace"].keys())
        for exp_id in exp_ids:
            d = exp_dict["stuff"]["trace"][exp_id]
            res_dict[exp_id] = [np.mean(d[str(nn)]) for nn in sorted(d, key=lambda x: int(x))]
        return res_dict

    if var_name == "Accuracy":
        res_dict = {}
        if exp_ids is None:
            exp_ids = list(exp_dict["stuff"]["acc"].keys())
        for exp_id in exp_ids:
            d = exp_dict["stuff"]["acc"][exp_id]
            res_dict[exp_id] = [d[str(nn)] for nn in sorted(d, key=lambda x: int(x))]
        return res_dict

    runs = exp_dict["stuff"]["runs"]
    exp_res = {}

    for exp_id in runs:

        if (exp_ids is not None) and (exp_id not in exp_ids):
            continue

        if step == -1:
            try:
                last_step = max(runs[exp_id], key=lambda x: int(x))
                curr_step = last_step
            except:
                continue
        else:
            curr_step = step

        stop_trying = False
        res = None
        while not stop_trying:
            try:
                # going down the tree with node names given by var_name.split("/")
                curr_dict = runs[exp_id][curr_step]
                var_name_split = var_name.split("/")
                for n in var_name_split:
                    curr_dict = curr_dict[n]

                if "net" in curr_dict:
                    num_nets = int(max(curr_dict["net"], key=lambda x: int(x))) + 1  # +1 bc zero indexed
                    res = np.array([curr_dict["net"][str(nn)] for nn in range(num_nets)])
                    exp_res[exp_id] = res
                else:
                    res = curr_dict[""]
                    exp_res[exp_id] = res

                stop_trying = True
            except:
                if (step == -1) and (curr_step > last_step - 5):
                    curr_step -= 1
                else:
                    stop_trying = True
        if res is None:
            print("No {} for step {}".format(var_name, step))
            exp_res[exp_id] = None



    return exp_res
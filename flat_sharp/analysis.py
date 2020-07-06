import numpy as np
import pandas as pd
import pickle, os
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
import torch
from torch.utils.data import DataLoader
import sys

sys.path.append("..")
from nets import Nets
from utils import *

import re

from postprocessing import *
from lineages import *

import itertools


def get_end_stats(exp_dict):
    stuff = exp_dict["stuff"]
    stats_dict = {}

    if "runs" in stuff:
        runs = stuff["runs"]

    else:
        runs = None

    if "trace" in stuff:
        trace = stuff["trace"]  # assume the trace i get is from the end.
    else:
        trace = None

    if "acc" in stuff:
        accs = stuff["acc"]
    else:
        accs = None

    if "loss" in stuff:
        loss = stuff["loss"]
    else:
        loss = None

    if "dist" in stuff:
        dist = stuff["dist"]
    else:
        dist = None

    configs = stuff["configs"]
    for exp_id in configs.index:

        num_nets = configs.loc[exp_id]["num_nets"]
        if runs is not None:
            try:
                num_steps = max(runs[exp_id], key=lambda x: int(x)) - 1
            except:
                continue

        try:
            stats_dict[str(exp_id)] = {}

            if loss is None:
                Loss_train_list = [runs[exp_id][num_steps]["Loss"]["train"]["net"][str(nn)] for nn in range(num_nets)]
            else:
                Loss_train_list = [loss[exp_id][str(nn)][0] for nn in range(num_nets)]
                Loss_test_list = [loss[exp_id][str(nn)][1] for nn in range(num_nets)]

                stats_dict[str(exp_id)]["Loss Test Mean"] = np.mean(Loss_test_list)
                stats_dict[str(exp_id)]["Loss Test Max"] = np.max(Loss_test_list)
                stats_dict[str(exp_id)]["Loss Test Min"] = np.min(Loss_test_list)

            stats_dict[str(exp_id)]["Loss Train Mean"] = np.mean(Loss_train_list)
            stats_dict[str(exp_id)]["Loss Train Max"] = np.max(Loss_train_list)
            stats_dict[str(exp_id)]["Loss Train Min"] = np.min(Loss_train_list)

            if accs is None:
                Acc_test_list = [runs[exp_id][num_steps]["Accuracy"]["net"][str(nn)] for nn in range(num_nets)]
            else:
                Acc_train_list = [accs[exp_id][str(nn)][0] for nn in range(num_nets)]
                Acc_test_list = [accs[exp_id][str(nn)][1] for nn in range(num_nets)]

                stats_dict[str(exp_id)]["Acc Train Mean"] = np.mean(Acc_train_list)
                stats_dict[str(exp_id)]["Acc Train Max"] = np.max(Acc_train_list)
                stats_dict[str(exp_id)]["Acc Train Min"] = np.min(Acc_train_list)

            stats_dict[str(exp_id)]["Acc Test Mean"] = np.mean(Acc_test_list)
            stats_dict[str(exp_id)]["Acc Test Max"] = np.max(Acc_test_list)
            stats_dict[str(exp_id)]["Acc Test Min"] = np.min(Acc_test_list)

            if accs is not None:
                stats_dict[str(exp_id)]["Gap Mean"] = stats_dict[str(exp_id)]["Acc Test Mean"] - \
                                                      stats_dict[str(exp_id)]["Acc Train Mean"]

            if ("runs" in stuff) and ("resampling_idxs" in exp_dict):
                # get mean total path weigth
                Y_axis_name = "Potential/curr"

                x_vals, y_vals = get_runs_arr(exp_dict, Y_axis_name, exp_ids=[exp_id], is_mean=False)

                try:
                    sampling_arr = exp_dict["resampling_idxs"][exp_id]
                    resampling_arr = np.array([sampling_arr[str(i)] for i in range(len(sampling_arr))])[1:-1]

                    curr_lineage, curr_assignments = find_lineages(resampling_arr)
                    Ys = get_linages_vals(curr_lineage, y_vals[0])
                except:
                    Ys = y_vals[0].T
                    print("Did not use lineages for {}".format(exp_id))
                stats_dict[str(exp_id)]["Path Weight Sum"] = np.mean(np.sum(Ys, axis=1))

        except:
            print("Error: No stats for {}".format(exp_id))

        if dist is not None:
            stats_dict[str(exp_id)]["Dist Mean"] = np.mean(dist[exp_id])
            stats_dict[str(exp_id)]["Dist Max"] = np.max(dist[exp_id])
            stats_dict[str(exp_id)]["Dist Min"] = np.min(dist[exp_id])

        if trace is not None:

            try:
                Trace_list = [np.mean(trace[exp_id][str(nn)]) for nn in range(num_nets)]
                Trace_std_list = [np.std(trace[exp_id][str(nn)]) for nn in range(num_nets)]
                stats_dict[str(exp_id)]["Trace Mean"] = np.mean(Trace_list)
                stats_dict[str(exp_id)]["Trace Mean Std"] = np.mean(Trace_std_list)
                stats_dict[str(exp_id)]["Trace Max"] = np.max(Trace_list)
                stats_dict[str(exp_id)]["Trace Min"] = np.min(Trace_list)

                # print(dict_key)
                # print(trace[exp_id][str(0)])
                # print()

                # stats_dict[str(exp_id)]["Train Loss/Trace Correlation"] = get_correlation(Loss_train_list, Trace_list)
                # stats_dict[str(exp_id)]["Test Acc/Trace Correlation"] = get_correlation(Acc_test_list, Trace_list)
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


def _plot(plots, plots_names, X_axis_name, Y_axis_name, X_axis_bounds, Y_axis_bounds, X_axis_display_name=None, Y_axis_display_name=None, save_location=None):
    if len(plots_names) > 1:
        plt.legend(tuple(plots),
                   plots_names,
                   scatterpoints=1,
                   loc='best',
                   ncol=3,
                   fontsize=8)

    if X_axis_display_name is None:
        X_axis_display_name = X_axis_name
    if Y_axis_display_name is None:
        Y_axis_display_name = Y_axis_name

    plt.xlabel(X_axis_display_name)
    plt.ylabel(Y_axis_display_name)

    if X_axis_bounds is not None:
        plt.xlim(X_axis_bounds)
    if Y_axis_bounds is not None:
        plt.ylim(Y_axis_bounds)

    color = plt.cm.tab20(np.arange(12))
    mpl.rcParams['axes.prop_cycle'] = cycler('color', color) # plt.cm.Set3(np.arange(12))))
    if save_location is not None:
        plt.savefig(save_location + ".png")#, format='eps')
    plt.show()


def plot_stats(stats_pd, X_axis_name, Y_axis_name, Z_axis_name=None, filter_seperate=None, filter_not_seperate=None, save_exp_path=None, X_axis_bounds=None,
               Y_axis_bounds=None, X_axis_display_name=None, Y_axis_display_name=None):
    plots = []
    plots_names = []

    if (filter_seperate is None) and (filter_not_seperate is None):
        x_values = stats_pd[X_axis_name].to_numpy()
        y_values = stats_pd[Y_axis_name].to_numpy()

        plots.append(plt.scatter(x_values, y_values))
        plots_names.append("Plot all")
        if save_exp_path is not None:
            save_location = os.path.join(save_exp_path,
                                         "{}_{}_{}".format(X_axis_name, Y_axis_name.replace("/", "-"), str("all")))
        else:
            save_location = None

        _plot(plots, plots_names, X_axis_name, Y_axis_name, X_axis_bounds, Y_axis_bounds,
              X_axis_display_name=X_axis_display_name, Y_axis_display_name=Y_axis_display_name,
              save_location=save_location)
    else:
        if (("lr_bs_ratio" in filter_seperate) or ("lr_bs_ratio" in filter_not_seperate)) or (
                X_axis_name == "lr_bs_ratio") or (
                Y_axis_name == "lr_bs_ratio"):
            stats_pd["lr_bs_ratio"] = stats_pd["learning_rate"] / stats_pd["batch_train_size"]

        if filter_seperate is None:
            filter_seperate = []

        unique_seperate_filter_dict = {f: list(set(stats_pd[f])) for f in filter_seperate}
        unique_seperate_filter_keys = list(unique_seperate_filter_dict.keys())

        unique_not_seperate_filter_dict = {f: list(set(stats_pd[f])) for f in filter_not_seperate}
        unique_not_seperate_filter_keys = list(unique_not_seperate_filter_dict.keys())

        unique_all_filter_keys = unique_seperate_filter_keys + unique_not_seperate_filter_keys

        for s_comb in itertools.product(*unique_seperate_filter_dict.values()):

            for ns_comb in itertools.product(*unique_not_seperate_filter_dict.values()):
                comb = s_comb + ns_comb

                filter_pd = stats_pd[(stats_pd[unique_all_filter_keys] == comb).to_numpy().all(1)]

                x_values = filter_pd[X_axis_name].to_numpy()
                y_values = filter_pd[Y_axis_name].to_numpy()
                if Z_axis_name is not None:
                    y_values = filter_pd[Y_axis_name].to_numpy()

                plots.append(plt.scatter(x_values, y_values))
                plots_names.append(comb)

            if save_exp_path is not None:
                save_location = os.path.join(save_exp_path, "{}_{}_{}".format(X_axis_name, Y_axis_name.replace("/", "-"), str(s_comb)))
            else:
                save_location = None

            _plot(plots, plots_names, X_axis_name, Y_axis_name, X_axis_bounds, Y_axis_bounds,
                    X_axis_display_name=X_axis_display_name, Y_axis_display_name=Y_axis_display_name, save_location=save_location)

            plots = []
            plots_names = []



# def get_runs_plots_seperate(exp_dict, var_name="Kish", running_average_gamma=0.2):
#     exp_runs = exp_dict["stuff"]["runs"]
#     for i in exp_runs:
#
#         plot_list = [0]
#
#         print(i)
#
#         for step in sorted(exp_runs[i], key=lambda x: int(x)):
#             # try:
#             # going down the tree with node names given by var_name.split("/")
#             curr_dict = exp_runs[i][step]
#             var_name_split = var_name.split("/")
#             for n in var_name_split:
#                 curr_dict = curr_dict[n]
#
#             if "net" in curr_dict:
#                 num_nets = int(max(curr_dict["net"], key=lambda x: int(x))) + 1  # +1 bc zero indexed
#                 to_append = np.array([curr_dict["net"][str(nn)] for nn in range(num_nets)])
#
#             else:
#                 to_append = curr_dict[""]
#             to_append = plot_list[-1] * (1 - running_average_gamma) + running_average_gamma * to_append
#             plot_list.append(to_append)
#             # except:
#             #     print("No {} for step {}".format(var_name, step))
#         plt.plot(plot_list[1:])
#         plt.show()


def get_runs_arr(exp_dict, var_name="Kish", exp_ids=None, running_average_gamma=1, is_mean=False):
    exp_runs = exp_dict["stuff"]["runs"]
    all_arrs = []
    val_steps = []

    for i in exp_runs:

        curr_arr = None
        curr_val_steps = None
        if (exp_ids is not None) and (i not in exp_ids):
            continue

        for step in sorted(exp_runs[i], key=lambda x: int(x)):
            try:

                # going down the tree with node names given by var_name.split("/")
                curr_dict = exp_runs[i][step]
                var_name_split = var_name.split("/")
                for n in var_name_split:
                    curr_dict = curr_dict[n]
                if "net" in curr_dict:
                    num_nets = int(max(curr_dict["net"], key=lambda x: int(x))) + 1  # +1 bc zero indexed

                    to_append = np.array([[curr_dict["net"][str(nn)] for nn in range(num_nets)]])
                    if is_mean:
                        to_append = np.mean(to_append).reshape(1, -1)
                else:
                    to_append = np.array([curr_dict[""]])

                if curr_arr is None:
                    curr_arr = to_append
                else:
                    to_append = curr_arr[-1] * (1 - running_average_gamma) + running_average_gamma * to_append
                    curr_arr = np.concatenate((curr_arr, to_append), axis=0)

                if curr_val_steps is None:
                    curr_val_steps = [step]
                else:
                    curr_val_steps.append(step)
            except:
                pass
                # print("No {} for step {}".format(var_name, step))

        all_arrs.append(curr_arr)
        val_steps.append(curr_val_steps)

    return val_steps, np.array(all_arrs)

def get_stat_step(exp_dict, var_name, step, exp_ids=None):
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


def get_mod_idx(arr, mod):
    if mod == "all":
        return list(range(len(arr)))

    if mod == "max":
        idx = np.argmax(arr)
    else:
        idx = np.argmin(arr)
    return [idx]


def get_selector_mod(exp_dict, axis_name):
    name_mod_split = axis_name.split(":")
    if len(name_mod_split) == 2:
        name, mod = name_mod_split
    else:
        name, mod = axis_name, None

    name_split = name.split(" ")

    tmp_cache = {}

    def helper(exp_id, nn_idx):
        if name_split[0] ==  "path":
            if exp_id not in tmp_cache:
                # get mean total path weigth
                Y_axis_name = "Potential/curr"

                x_vals, y_vals = get_runs_arr(exp_dict, Y_axis_name, exp_ids=[exp_id], is_mean=False)
                print(y_vals.shape)
                try:
                    sampling_arr = exp_dict["resampling_idxs"][exp_id]
                    resampling_arr = np.array([sampling_arr[str(i)] for i in range(len(sampling_arr))])[1:-1]

                    curr_lineage, curr_assignments = find_lineages(resampling_arr)
                    inverted_curr_assignments = {}
                    for k, v in curr_assignments.items():
                        for vv in v:
                            inverted_curr_assignments[vv] = k
                    Ys = get_linages_vals(curr_lineage, y_vals[0])
                    sum_ys = [np.sum(Ys[inverted_curr_assignments[nn]]) + y_vals[0][-1][nn] for nn in range(y_vals.shape[2])]

                except:
                    Ys = y_vals[0].T
                    # print("Did not use lineages for {}".format(exp_id))
                    sum_ys = [np.sum(Ys[nn]) for nn in range(y_vals.shape[2])]

                tmp_cache[exp_id] = sum_ys

            else:
                return tmp_cache[exp_id][nn_idx]
        #     stats_dict[str(exp_id)]["Path Weight Sum"] = np.mean(np.sum(Ys, axis=1))
        elif name_split[0] == "eigs":
            eigs = exp_dict["stuff"]["eig"][exp_id][str(nn_idx)]
            if name_split[1] == "min":
                return min(eigs)
            else:
                return max(eigs)
        elif name_split[0] in exp_dict["stuff"]["configs"].loc[exp_id]:
            return exp_dict["stuff"]["configs"].loc[exp_id][name_split[0]]
        elif name_split[0] == "grad":
            return exp_dict["stuff"]["grad"][exp_id][str(nn_idx)]
        elif name_split[0] == "trace":
            return np.mean(exp_dict["stuff"]["trace"][exp_id][str(nn_idx)])
        elif name_split[0] == "gap":
            if name_split[1] == "acc":
                d = exp_dict["stuff"]["acc"][exp_id][str(nn_idx)]
            else:
                d = exp_dict["stuff"]["loss"][exp_id][str(nn_idx)]
            return d[1] - d[0]

        else:
            xs = exp_dict["stuff"][name_split[0]][exp_id][str(nn_idx)]
            if name_split[1] == "train":
                return xs[0]
            else:
                return xs[1]

    return helper, mod


def get_plot_special(exp_dict, exp_ids, X_axis_name, Y_axis_name):
    X_selector, X_mod = get_selector_mod(exp_dict, X_axis_name)
    Y_selector, Y_mod = get_selector_mod(exp_dict, Y_axis_name)

    assert not ((X_mod is not None) and (Y_mod is not None))
    assert (X_mod is not None) or (Y_mod is not None)

    x_vals = []
    y_vals = []
    for exp_id in exp_ids:
        num_nets = exp_dict["stuff"]["configs"].loc[exp_id]["num_nets"]
        Xs = [X_selector(exp_id, i) for i in range(num_nets)]
        Ys = [Y_selector(exp_id, i) for i in range(num_nets)]

        if X_mod is not None:
            nn_idxs = get_mod_idx(Xs, X_mod)
        else:
            nn_idxs = get_mod_idx(Ys, Y_mod)

        if (len(nn_idxs) == 0):
            continue

        x_vals.append([X_selector(exp_id, i) for i in nn_idxs])
        y_vals.append([Y_selector(exp_id, i) for i in nn_idxs])

    return np.array(x_vals).reshape(-1), np.array(y_vals).reshape(-1)


def plot_special(exp_dict, X_axis_name, Y_axis_name, filter_seperate=None, filter_not_seperate=None,
                 save_exp_path=None, X_axis_bounds=None, Y_axis_bounds=None, pre_filtered_exp_ids=None, is_mean=False,
                 X_axis_display_name=None, Y_axis_display_name=None):
    plots = []
    plots_names = []

    cfs = exp_dict["stuff"]["configs"]

    if (filter_seperate is None) and (filter_not_seperate is None):
        x_vals, y_vals = get_plot_special(exp_dict, list(cfs.index), X_axis_name, Y_axis_name)
        plots.append(plt.scatter(x_vals, y_vals))
        plots_names.append("Plot all")
        if save_exp_path is not None:
            save_location = os.path.join(save_exp_path,
                                         "{}_{}_{}".format(X_axis_name, Y_axis_name.replace("/", "-"), str("all")))
        else:
            save_location = None

        _plot(plots, plots_names, X_axis_name, Y_axis_name, X_axis_bounds, Y_axis_bounds,
              X_axis_display_name=X_axis_display_name, Y_axis_display_name=Y_axis_display_name,
              save_location=save_location)
    else:
        if filter_seperate is None:
            filter_seperate = []

        unique_seperate_filter_dict = {f: list(set(cfs[f])) for f in filter_seperate}
        unique_seperate_filter_keys = list(unique_seperate_filter_dict.keys())

        unique_not_seperate_filter_dict = {f: list(set(cfs[f])) for f in filter_not_seperate}
        unique_not_seperate_filter_keys = list(unique_not_seperate_filter_dict.keys())

        unique_all_filter_keys = unique_seperate_filter_keys + unique_not_seperate_filter_keys

        for s_comb in itertools.product(*unique_seperate_filter_dict.values()):

            for ns_comb in itertools.product(*unique_not_seperate_filter_dict.values()):
                comb = s_comb + ns_comb

                exp_ids = list(cfs[(cfs[unique_all_filter_keys] == comb).to_numpy().all(1)].index)

                if pre_filtered_exp_ids is not None:
                    exp_ids = list(set(exp_ids) & set(pre_filtered_exp_ids))

                if len(exp_ids) == 0:
                    continue

                if X_axis_name == "time":
                    x_vals, y_vals = get_runs_arr(exp_dict, Y_axis_name, exp_ids, is_mean=False)

                    plot_y_vals = get_exp_lineages(exp_dict, x_vals, y_vals, exp_ids, is_mean=is_mean)
                    print(x_vals)
                    plot_x_vals = np.array([x_vals[0] for _ in range(len(plot_y_vals))])
                    plots.append(plt.plot(plot_x_vals.T, plot_y_vals.T)[0])

                else:
                    x_vals, y_vals = get_plot_special(exp_dict, exp_ids, X_axis_name, Y_axis_name)

                    print("Correlation for {} {}/{}: {}".format(comb, X_axis_name, Y_axis_name,
                                                                get_correlation(x_vals, y_vals)))

                    plots.append(plt.scatter(x_vals, y_vals))
                plots_names.append(comb)
            if save_exp_path is not None:
                save_location = os.path.join(save_exp_path, "{}_{}_{}".format(X_axis_name, Y_axis_name.replace("/", "-"), str(s_comb)))
            else:
                save_location = None

            _plot(plots, plots_names, X_axis_name, Y_axis_name, X_axis_bounds, Y_axis_bounds,
                    X_axis_display_name=X_axis_display_name, Y_axis_display_name=Y_axis_display_name, save_location=save_location)

            plots = []
            plots_names = []

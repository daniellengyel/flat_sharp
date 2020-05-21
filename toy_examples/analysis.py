import numpy as np
import pandas as pd
import os, pickle, yaml
from postprocessing import *
import itertools



def _count_basins(attractive_basins, all_paths, N):
    """Assumes tau (used for resampling) is larger than N"""
    basin_counter = {}
    old_shape = all_paths.shape
    reshaped_all_paths = np.transpose(all_paths, (0, 2, 1, 3)).reshape([old_shape[0]*old_shape[2], old_shape[1], old_shape[3]])
    for i, basin in enumerate(attractive_basins):
        total_curr_count = 0
        for t in range(N):
            curr_paths = reshaped_all_paths[-(t + 1)].reshape(-1)
            total_curr_count += sum(1*((basin[0] < curr_paths) & (curr_paths <= basin[1])))
        basin_counter[str(i)] = total_curr_count/float(N)
    return basin_counter


def get_count_basins(exp_folder, attractive_basins):
    count_dir = {}

    for root, dirs, files in os.walk("{}".format(exp_folder), topdown=False):
        if not os.path.isfile(os.path.join(root, "results.pkl")):
            continue

        curr_dir = os.path.basename(root)
        with open(os.path.join(root, "results.pkl"), "rb") as f:
            all_paths = pickle.load(f)
        count_dir[curr_dir] = _count_basins(attractive_basins, all_paths, 4)  # 4 is arbitrary for N

    count_pd = pd.DataFrame(count_dir).T
    count_pd["total_in_basins"] = count_pd.sum(axis=1)

    return count_pd


# get files... iterate through all paths

def get_configs(exp_folder):
    config_dir = {}
    for root, dirs, files in os.walk("{}".format(exp_folder), topdown=False):
        if not os.path.isfile(os.path.join(root, "config.yml")):
            continue

        curr_dir = os.path.basename(root)
        with open(os.path.join(root, "config.yml"), "rb") as f:
            config = yaml.load(f)
        config_dir[curr_dir] = config

    return pd.DataFrame(config_dir).T

# def get_varied_hp(configs_df):

# analysis
def get_stats(exp_folder, attractive_basins):
    count_pd = get_count_basins(exp_folder, attractive_basins)
    cfs = get_configs(exp_folder)

    for i in range(len(attractive_basins)):
        count_pd["{}_per".format(i)] = count_pd["{}".format(i)] / cfs["num_particles"]
    cfs_hp = get_hp(cfs)
    cfs_hp_df = cfs[list(cfs_hp.keys())]
    stats_pd = pd.concat([count_pd, cfs_hp_df], axis=1)

    return stats_pd, cfs



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


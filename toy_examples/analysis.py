import numpy as np
import pandas as pd
import os, pickle, yaml
from postprocessing import *


def _count_basins(attractive_basins, all_paths, N):
    """Assumes tau (used for resampling) is larger than N"""
    basin_counter = {}
    for i, basin in enumerate(attractive_basins):
        total_curr_count = 0
        for t in range(N):
            curr_paths = all_paths[-1][:, -(t + 1)].reshape(-1)
            total_curr_count += sum(1*((basin[0] <= curr_paths) & (curr_paths <= basin[1])))
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
    for i in range(len(attractive_basins)):
        count_pd["{}_per".format(i)] = count_pd[str(i)] / count_pd["total_in_basins"]

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
        count_pd["{}_per".format(i)] *= count_pd["total_in_basins"] / cfs["num_particles"]
    cfs_hp = get_hp(cfs)
    cfs_hp_df = cfs[list(cfs_hp.keys())]
    stats_pd = pd.concat([count_pd, cfs_hp_df], axis=1)

    return stats_pd, cfs


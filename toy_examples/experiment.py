import numpy as np
import matplotlib.pyplot as plt
from ray import tune


import sys, os
import pickle
import yaml

from Saving import *
from Optimizations import *
from postprocessing import *


def experiment_run(config_inp, path):
    # run experiment

    # save the results and process
    file_stamp = str(time.time())
    exp_folder = os.path.join(path, file_stamp)
    os.mkdir(exp_folder)

    results, path_weights = diffusion_resampling(config_inp, exp_folder)


    with open(os.path.join(exp_folder, "config.yml"), "w") as f:
        yaml.dump(config_inp, f, default_flow_style=False)
    with open(os.path.join(exp_folder, "results.pkl"), "wb") as f:
        pickle.dump(results, f)
    with open(os.path.join(exp_folder, "path_weights.pkl"), "wb") as f:
        pickle.dump(path_weights, f)

# g_params = [[-10, 1, -2], [0, 10, -1], [10, 3, 1], [-20, 1, 3], [20, 1, 3]]
# g_params = [[-10, 0.7, -1], [0, 10, -1.5], [9, 3, 1], [17.5, 10, -0.5], [-20, 1, 3], [25, 1, 3]]
# g_params = [[-40, 0.7, -2],[-30, 0.7, -1.5],[-5, 0.7, -1], [35, 0.7, -1.5],
#             [-17.5, 30, -3], [5, 10, -1.5], [14, 3, 1], [25, 10, -0.5],
#             [-50, 1, 3], [50, 1, 3]]

# g_params = [[-70, 0.7, -2], [-55, 14, -8], [-40, 0.7, -2], [-25, 14, -8], [-10, 0.7, -2],
#             [5, 14, -8],  [20, 0.7, -2], [35, 14, -8], [50, 0.7, -2],
#             [-85, 1, 3], [65, 1, 3]] # flat, sharp, flat, sharp, ...

# g_params = [[-17.5, 10, -6.5], [0, 2, -3], [17.5, 10, -6.5], [-35, 31, 3], [35, 1, 3]] # symmetric

# g_params = [[-15, 0.5, 1], [0.5, 0.7, 1], [0, 15, -10], [11.5, 20, -10.5], [25, 0.5, 1]]
# g_params = [[-15, 0.5, 1],  [0.5, 0.2, 0.5], [0, 25, -14],  [5.5, 0.2, 0.4], [11.5, 20, -11.2], [25, 0.5, 1]]
# g_params = [[-15, 0.5, 1],  [0.5, 0.2, 0.2], [0, 40, -20],  [5.5, 0.2, 1], [13, 20, -12], [25, 0.5, 1]]
# g_params = [[-15, 0.5, 1],  [0.5, 0.2, 0.2], [0, 40, -20],  [5.5, 0.2, 1], [13, 20, -12.2], [25, 0.5, 1]]
# g_params = [[-20, 0.5, 1], [-9, 0.5, -3], [0.5, 0.2, 0.5], [0, 40, -20.25],  [5.5, 0.2, 1], [13, 20, -12.2], [30, 0.5, 1]]
# g_params = [[-25, 0.5, 1], [0, 7, -4.5],  [-7, 7, -4.5],  [-14, 7, -4.5],
#             [13, 20, -8], [28, 0.5, 1]]
# g_params = [[-37, 0.5, 1], [-27, 15, -2], [0, 7, -4.5],  [-7, 7, -4.5],  [-14, 7, -4.5],
#             [13, 20, -8], [30, 15, -2], [40, 0.5, 1]]

# simple
# g_params = [[-17.5, 1, -2], [17.5, 10, -6.5], [-2, 15, -2],[ -35, 1, 3], [35, 1, 3]]

# complex gaussian
g_params = [[-54, 0.5, 1], [-44, 15, -2], [-30, 4.5, -3.8], [-24, 4.5, -3.8], [-18, 4.5, -3.8], [0, 4.5, -3.6],  [-6, 4.5, -3.8],  [-12, 4.5, -3.8],
            [12, 30, -10.1], [35, 15, -2], [45, 0.5, 1]]




np_g_params = np.array(g_params)

process = {}

process["particle_init"] = "1d_uniform"
process["num_particles"] = tune.grid_search([100])

process["potential_function"] = "gaussian"
process["potential_function_params"] = g_params

process["regularizer"] = False # "1d_hessian_trace"
process["regularizer_lambda"] = 0 # tune.grid_search([-100, -10, -5, 0, 1, 5, 10, 100])

process["local_entropy"] = False
process["local_entropy_exact"] = False
process["local_entropy_L"] = 0 # tune.grid_search([5, 10])
process["local_entropy_gamma"] = 0 # tune.grid_search(list(np.linspace(3, 5, 5)))
process["local_entropy_alpha"] = 0 # 0.75
process["local_entropy_step_size"] = 0.1 # tune.grid_search([0.01, 0.1, 0.5, 1]) # use 1 for exact method
process["local_entropy_var"] = 0 # tune.grid_search([0.0001, 0.001, 0.01])

process["total_iter"] = 20 # calling this total iter is misleading. It is the total number of times that we resample
process["tau"] = tune.grid_search([10])
process["x_range"] = [min(np_g_params[:, 0]), max(np_g_params[:, 0])]

process["learning_rate"] = 41.1111	 # tune.grid_search(list(np.linspace(10, 80, 10)))
process["temperature"] = 0.453333 #  tune.grid_search(list(np.linspace(0.01, 4, 10))) # 0.2
process["epsilon"] = 0

process["weight_function"] = "norm"
process["weight_function_params"] = None
process["softmax_beta"] = tune.grid_search(list(np.linspace(-4, 4, 20)))

process["domain_enforcer"] = "hyper_cube_enforcer"
process["domain_enforcer_params"] = 0.2

process["seed"] = 100
process["return_full_path"] = True

# --- Set up folder in which to store all results ---
folder_name = get_file_stamp()
cwd = os.environ["PATH_TO_TOY_FOLDER"]
folder_path = os.path.join(cwd, "experiments", process["potential_function"], folder_name)
print(folder_path)
os.makedirs(folder_path)
os.mkdir(os.path.join(folder_path, "runs"))

save_plot({"x_range": process["x_range"], "potential_function": process["potential_function"], "potential_function_params": process["potential_function_params"]}, folder_path)

# diffusion_resampling(process, folder_path, return_full_path=True)

msg = "Use FSM on the symmetric function."
with open(os.path.join(folder_path, "description.txt"), "w") as f:
    f.write(msg)

analysis = tune.run(lambda config_inp:  experiment_run(config_inp, os.path.join(folder_path, "runs")), config=process)
print(analysis)
# create_animations(folder_path)

# TODO have logging of what we want to achieve with the current experiment.
# add a new distance metric. Distance from permutations -- used to know how many networks are needed.
# could also tell us how far away the vallys are and how symmetric the space is.
# Repeat experiments in finding minima with SGD paper

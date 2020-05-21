import autograd.numpy as np
from Kernels import *
from utils import *
from Functions import *

from Functions import Gibbs, GradGibbs

import time

def diffusion_resampling(process, path, verbose=False, domain_enforcer=None):
    """Returns the paths of the particles in the format:
    num_resample_steps, num_particles, tau + 1, dim.
    We get tau+1 because of the resampling step."""
    if process["seed"] is not None:
        np.random.seed(process["seed"])

    p_start = get_particles(process)
    p_gamma = lambda t: process["learning_rate"]
    p_temperature = lambda t: process["temperature"]
    p_num_particles = len(p_start)
    p_epsilon = process["epsilon"]
    total_iter, tau = process["total_iter"], process["tau"]
    if tau is None:
        tau = total_iter
        num_resample_steps = 1
    else:
        assert total_iter % tau == 0
        num_resample_steps = int(total_iter / tau)
    return_full_path = process["return_full_path"]

    dim = len(p_start)

    # get potential_function and gradient
    U, grad_U = get_potential(process)

    # get weight_function
    p_weight_func = get_weight_function(process)

    # get resample_function
    p_resample_func = get_resample_function(process)

    # get domain_enforcer
    x_range = process["x_range"]
    if process["domain_enforcer"] == "hyper_cube_enforcer":
        domain_enforcer_strength = process["domain_enforcer_params"]
        domain_enforcer = hyper_cube_enforcer(x_range[0], x_range[1], domain_enforcer_strength)
    else:
        raise ValueError("Does not support given function {}".format(process["domain_enforcer"]))

    # TODO init saving
    # file_stamp = str(time.time())  # get_file_stamp()
    # writer = SummaryWriter("{}/runs/{}".format(folder_path, file_stamp))

    # init num_particles
    all_paths = []
    p_weights = np.array([[0] for _ in p_start])
    curr_paths = np.array([[np.array(p)] for p in p_start])
    all_weights = []

    for t in range(num_resample_steps):
        for t_tau in range(tau):
            # --- diffusion step ---
            x_curr = np.array(curr_paths[:, -1])
            x_next = x_curr + p_gamma(t) * (
                -grad_U(x_curr.T).T) + p_temperature(t) * np.random.normal(size=x_curr.shape)
            if domain_enforcer is not None:
                x_next, went_outside_domain = domain_enforcer(x_next)
            # ----

            # weight update
            weights_next = p_weight_func(U, grad_U, x_curr.T,
                                      p_weights[:, -1])  # Todo use the computed grads for update step here instead of recomupting. should make everything twice as fast

            if return_full_path or (curr_paths.shape[1] == 1):
                curr_paths = np.concatenate([curr_paths, x_next.reshape([curr_paths.shape[0], 1, curr_paths.shape[2]])], axis=1)
                p_weights = np.concatenate([p_weights, weights_next.reshape([p_weights.shape[0], 1])], axis=1)
            else:
                curr_paths[:, -1] = x_next.reshape([curr_paths.shape[0], curr_paths.shape[2]])
                p_weights[:, -1] = weights_next.reshape([p_weights.shape[0]])


        # add paths and weights
        all_weights.append(p_weights)
        all_paths.append(curr_paths)
        end_points = curr_paths[:, -1]

        # resample particles
        new_starting = list(p_resample_func(p_weights[:, -1], end_points))
        curr_paths = np.array([[p] for p in new_starting])

        p_weights = np.array([[0] for _ in p_start])

    return np.array(all_paths), np.array(all_weights)





def grad_descent(func, grad_func, x_curr, eps, gamma, start_t=0, end_t=float("inf"), verbose=False, domain_enforcer=None):
    x_curr = np.array(x_curr, dtype=np.float)
    if domain_enforcer is not None:
        x_curr, went_outside_domain = domain_enforcer(x_curr)
    path = [x_curr]
    t = start_t

    went_outside_domain = False
    while t < end_t:
        x_next = x_curr - gamma(t) * grad_func(x_curr)
        if domain_enforcer is not None:
            x_next, went_outside_domain = domain_enforcer(x_next)

        path.append(x_next)

        if (np.linalg.norm(x_next - x_curr)) < eps and not went_outside_domain:  # TODO check what happens with more samples
            if verbose:
                print(grad_func(x_curr))
            break
        if (t % 50) == 0 and verbose:
            print("Iteration", t)
            print("diff", np.abs(func(x_next) - func(x_curr)))
        x_curr = x_next
        t += 1
    return np.array(path)



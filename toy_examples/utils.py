import autograd.numpy as np
from Functions import *
from autograd import jacobian
from scipy.stats import entropy


# ------
# Process util stuff
def get_potential(process):
    # get potential_function and gradient
    if process["potential_function"] == "gaussian":
        potential_params = process["potential_function_params"]
        U = gaussian_sum(potential_params)
        grad_U = grad_gaussian_sum(potential_params)
    elif process["potential_function"] == "Ackley":
        U = AckleyProblem
        grad_U = GradAckleyProblem
    elif process["potential_function"] == "2d_gaussian_symmetric":
        potential_params = process["potential_function_params"]
        one_d_U = gaussian_sum(potential_params)
        one_d_grad_U = grad_gaussian_sum(potential_params)
        U = lambda inp: one_d_U(np.array([inp[0]])) + one_d_U(np.array([inp[1]]))
        grad_U = lambda inp: np.array([one_d_grad_U(np.array([inp[0]]))[0],  one_d_grad_U(np.array([inp[1]]))[0]])
    elif process["potential_function"] == "2d_gaussian_symmetric_attraction":
        params = process["potential_function"]["params"]
        one_d_U = gaussian_sum(params["g_params"])
        one_d_grad_U = grad_gaussian_sum(params["g_params"])
        alpha, p = params["attraction"]["alpha"], params["attraction"]["origin_point"]
        U = lambda inp: one_d_U(np.array([inp[0]])) + one_d_U(np.array([inp[1]])) + alpha * (
                    (inp[0] - p[0]) ** 2 + (inp[1] - p[1]) ** 2)
        grad_U = lambda inp: np.array([one_d_grad_U(np.array([inp[0]]))[0] + 2 * alpha * (inp[0] - p[0]),
                                       one_d_grad_U(np.array([inp[1]]))[0] + 2 * alpha * (inp[1] - p[1])])
    else:
        raise ValueError("Does not support given function {}".format(process["potential_function"]))
    if ("regularizer" in process) and (process["regularizer"] is not None):
        if process["regularizer"] == "1d_hessian_trace":
            def reg_helper(process, f, grad_f):
                reg_lambda = process["regularizer_lambda"]
                potential_params = process["potential_function_params"]
                reg_f = lambda inp: f(inp) + reg_lambda*trace_hessian_gaussian_sum(potential_params)(inp)
                reg_grad_f = lambda inp: grad_f(inp) + reg_lambda*grad_trace_hessian_gaussian_sum(potential_params)(inp)
                return reg_f, reg_grad_f
            U, grad_U = reg_helper(process, U, grad_U)

    if ("local_entropy" in process) and (process["local_entropy"]):
        if process["local_entropy_exact"]:
            def local_entropy_exact_helper(process, grad_f, f):
                gamma, SGDL_step_size = process["local_entropy_gamma"], process["local_entropy_step_size"]
                return lambda inp: approx_local_entroy_grad(inp, f, SGDL_step_size, gamma)
            grad_U = local_entropy_exact_helper(process, grad_U, U)

        else:
            def local_entropy_helper(process, grad_f, f):
                L, gamma, alpha, SGDL_step_size, var = process["local_entropy_L"], process["local_entropy_gamma"], process["local_entropy_alpha"], process["local_entropy_step_size"], process["local_entropy_var"]
                return lambda inp: SGDL(inp, L, gamma, alpha, SGDL_step_size, var, grad_f, f)
            grad_U = local_entropy_helper(process, grad_U, U)
    return U, grad_U


def get_particles(process):
    # get start_pos
    if process["particle_init"] == "1d_uniform":
        num_particles = process["num_particles"]
        particles = [[np.random.uniform(process["x_range"][0], process["x_range"][1])] for _ in range(num_particles)]
    elif process["particle_init"] == "2d_uniform": # for now same range for all dimensions
        num_particles = process["num_particles"]
        x_low, x_high = process["x_range"]
        particles = [[np.random.uniform(x_low, x_high), np.random.uniform(x_low, x_high)] for _ in range(num_particles)]
    elif process["particle_init"] == "2d_position": # for now same range for all dimensions
        num_particles = process["num_particles"]
        x_low, x_high = process["x_range"]
        p = np.array(process["particle_init"]["params"]["position"])
        assert ((x_low <= p) & (p <= x_high)).all()
        particles = [p for _ in range(num_particles)]
    else:
        raise ValueError("Does not support given function {}".format(process["particle_init"]))
    return np.array(particles)

def get_resample_function(process):
    if (process["softmax_beta"] is not None) and (process["softmax_beta"] != 0):
        resample_beta = process["softmax_beta"]
        p_resample_func = lambda w, end_p: resample_positions_softmax(w, end_p, beta=resample_beta)
    else:
        p_resample_func = lambda w, end_p: end_p
    return p_resample_func

def get_weight_function(process):
    if process["weight_function"] == "norm":
        p_weight_func = lambda U, grad_U, x, curr_weights: weight_function_discounted_norm(U, grad_U, x, curr_weights, 1)
    elif process["weight_function"] == "discounted_norm":
        weight_gamma = process["weight_function"]["params"]["gamma"]
        p_weight_func = lambda U, grad_U, x, curr_weights: weight_function_discounted_norm(U, grad_U, x, curr_weights,
                                                                                             weight_gamma)
    elif process["weight_function"] == "partial_norm":
        partials = process["weight_function"]["params"]["partials"]
        p_weight_func = lambda U, grad_U, x, curr_weights: weight_function_discounted_norm(U, grad_U, x, curr_weights,
                                                                                           partials=partials)
    else:
        raise ValueError("Does not support given function {}".format(process["weight_function"]))
    return p_weight_func


def get_num_steps(process):
    x_low, x_high, t_low, t_high = process["domain"]
    num_x, num_t = (x_high - x_low) / float(delta_x), (t_high - t_low) / float(delta_t)
    num_x, num_t = int(num_x) + 1, int(num_t) + 1
    return num_x, num_t

def get_init_density(process):
    # get start_pos
    if process["density_init"]["name"] == "uniform":
        delta_x = process["delta_x"]
        num_x = get_num_steps(process)[0]
        return np.array([[u_start(delta_x * i) for i in range(num_x)]])
    else:
        raise ValueError("Does not support given function {}".format(process["density_init"]["name"]))

# -------
# diffusion stuff
def resample_positions_softmax(weights, positions, beta=1):
    probabilities = softmax(weights, beta)
    pos_filter = np.random.choice(list(range(len(positions))), len(positions), p=probabilities)
    return np.array(positions)[np.array(pos_filter)]

def softmax(weights, beta=-1):
    # normalize weights:
    weights /= np.sum(weights)
    sum_exp_weights = sum([np.exp(beta * w) for w in weights])
    probabilities = np.array([np.exp(beta * w) for w in weights]) / sum_exp_weights
    return probabilities

def weight_function_discounted_norm(U, grad_U, x, curr_weights, gamma=1, partials=None):
    grad = grad_U(x)
    if partials is not None:
        grad = grad[partials]
    return gamma * curr_weights + np.linalg.norm(grad, axis=0)

def kish_effs(weights):
    """Assume weights are just a list of numbers"""
    N = len(weights)
    weights = np.array(weights)
    sum_weights = np.sum(weights)
    return 1/float(N) *  sum_weights**2 / weights.dot(weights)


#define potential for second proccess
def U_second(U, k, kernel, particles):
    def U_second_helper(x):
        return U(x) + k*V(x, kernel, particles)
    return U_second_helper


def grad_U_second(grad_U, k, grad_kernel, particles):
    return U_second(grad_U, k, grad_kernel, particles)


# Approximating density with the particles
def V(x, K, particles):
    N = len(particles)
    ret_sum = 0
    for p in particles:
        ret_sum += K(x, p)
    return 1 / float(N) * ret_sum


def grad_V(x, grad_K, particles):
    return V(x, grad_K, particles)

def particles_converged(p_paths, epsilon):
    for p in p_paths:
        if not ((len(p) > 2) and (np.linalg.norm(p[-1] - p[-2]) < epsilon)):
            return False
    return True

def hyper_cube_enforcer(lower_bound=-32.768, upper_bound=32.768, reflective_strength=1):
    def helper(x):
        filter_lower = x < lower_bound
        x[filter_lower] = lower_bound + reflective_strength

        filter_upper = x > upper_bound
        x[filter_upper] = upper_bound - reflective_strength
        return x, np.any(filter_lower) or np.any(filter_upper)
    return helper

def percent_endpoint(x_star, end_points, epsilon):
    num_reached = 0
    for p in end_points:
        if np.linalg.norm(x_star - p) < epsilon:
            num_reached += 1
    return num_reached * 1.0 / len(end_points)


def filter_to_goal(x_star, epsilon, analytics):
    filter_distance = np.array([np.linalg.norm(p - x_star) < epsilon for p in analytics["end_point"].values
                                ])

    return filter_distance


def get_grad(U):
    return lambda *args: np.sum(jacobian(U, 0)(*args), axis=-1)





def KL_symmetric_divergence(p1, p2, x_range, delta):
    bins = np.linspace(x_range[0], x_range[1], (x_range[1] - x_range[0]) / float(delta) + 1)
    p1_hist, p1_bin_edges = np.histogram(p1, bins)
    p2_hist, p2_bin_edges = np.histogram(p2, bins)

    p1_hist = p1_hist.astype('float64')
    p2_hist = p2_hist.astype('float64')

    p1_hist[p1_hist == 0] = 1e-25
    p2_hist[p2_hist == 0] = 1e-25
    return entropy(p1_hist, p2_hist, base=2) + entropy(p2_hist, p1_hist, base=2)


def kish_effs(weights):
    """Assume weights are just a list of numbers"""
    N = len(weights)
    weights = np.array(weights)
    sum_weights = np.sum(weights)
    return 1/float(N) *  sum_weights**2 / weights.dot(weights)


if __name__ == "__main__":
    from Kernels import *

    x = np.array([0])
    y = np.array([0, 1, 4, 65, 123, 65])
    cov = np.eye(y.shape[0])

    cov = np.eye(1)
    grad_k = grad_gaussian(cov)

    V(x, grad_k, [x])
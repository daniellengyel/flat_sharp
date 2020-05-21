from Functions import *
from Optimizations import *
from utils import *
from Saving import *

import shutil



def _reorder_paths_for_plots(all_paths, process):
    # want to get order: big_timestep, particle, tau_timestep, point

    f, grad_f = get_potential(process)

    inp = all_paths.reshape([np.product(all_paths.shape[:3]), all_paths.shape[-1]]).T
    d = inp.shape[0]
    out = f(inp)
    all_paths_proc = np.concatenate([inp.T, out.reshape([len(out), 1])], axis=1).reshape(all_paths.shape[0], all_paths.shape[1], all_paths.shape[2], d + 1)
    return all_paths_proc

def _get_animation(all_paths, process, process_id, root_animations_path):
    # TODO for only feynman kac paths etc
    all_paths_reordered = _reorder_paths_for_plots(all_paths, process)

    # get supplementary stuff
    d = all_paths_reordered.shape[-1] - 1
    f, grad_f = get_potential(process)
    ani_path = os.path.join(root_animations_path, process_id)
    os.makedirs(ani_path)

    if d == 1:
        # get the function plot inputs
        X = np.linspace(process["x_range"][0], process["x_range"][1], 200)
        inp = np.array([X])
        Y = f(inp)

        # full process densities
        K = multi_gaussian(np.array([[0.6]]))

        create_animation_1d_pictures_particles(all_paths_reordered, X, Y, ani_path,
                                                            graph_details={"p_size": 3,  # "density_function": None})
                                                                           "density_function":
                                                                               lambda x, p: V(np.array([x]), K, p)})


    elif d == 2:
        # get the function plot inputs

        X = np.linspace(process["x_range"][0], process["x_range"][1], 100)
        Y = np.linspace(process["x_range"][0], process["x_range"][1], 100)
        inp = np.array(np.meshgrid(X, Y)).reshape(2, len(X) * len(Y))

        Z = f(inp).reshape(len(X), len(Y))


        # full process densities
        K = multi_gaussian(np.array([[0.6, 0], [0, 0.6]]))

        create_animation_2d_pictures_particles(all_paths_reordered, X, Y, Z, ani_path,
                                                            graph_details={"type": "heatmap", "p_size": 3,
                                                                           # "density_function": None})
                                                                           "density_function": lambda inp, p: V(inp, K, p),
                                                                       "interpolation": "bilinear"})
    else:
        print(d)
        raise Exception("Error: Dimension {} does not exist.".format(d))

    print(ani_path)
    print(root_animations_path)

    create_animation(ani_path, os.path.join(root_animations_path, "{}.mp4".format(process_id)), framerate=10)

    time.sleep(3)  # otherwise it deletes the images before getting the video
    shutil.rmtree(ani_path)  # remove dir and all contains


def get_animation(exp_folder, process_id):
    root_animations_path = os.path.join(exp_folder, "animations")

    if not os.path.isdir(root_animations_path):
        os.mkdir(root_animations_path)

    runs_folder = os.path.join(exp_folder, "runs")

    id_run_folder = os.path.join("{}".format(runs_folder), process_id)

    with open(os.path.join(id_run_folder, "config.yml"), "r") as f:
        config = yaml.load(f)
    with open(os.path.join(id_run_folder, "results.pkl"), "rb") as f:
        all_paths = pickle.load(f)
    _get_animation(all_paths, config, process_id, root_animations_path)


def get_animations(exp_folder):
    root_animations_path = os.path.join(exp_folder, "animations")


    if not os.path.isdir(root_animations_path):
        os.mkdir(root_animations_path)

    runs_folder = os.path.join(exp_folder, "runs")

    for process_id in os.listdir(runs_folder):
        root = os.path.join("{}".format(runs_folder), process_id)

        if "DS_Store" in process_id:
            continue
        if not os.path.isdir(root):
            continue

        print(process_id)
        with open(os.path.join(root, "config.yml"), "r") as f:
            config = yaml.load(f)
        with open(os.path.join(root, "results.pkl"), "rb") as f:
            all_paths = pickle.load(f)
        _get_animation(all_paths, config, process_id, root_animations_path)

def get_config_to_id_map(configs):
    map_dict = {}

    for net_id in configs:
        conf = configs[net_id]
        tmp_dict = map_dict
        for k, v in conf.items():
            if "potential" in k:
                continue
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

def different_cols(df):
    a = df.to_numpy() # df.values (pandas<0.24)
    a[:, 14] = 0
    return (a[0] != a[1:]).any(0)

def get_hp(cfs):
    filter_cols = different_cols(cfs)
    hp_names = cfs.columns[filter_cols]
    hp_dict = {hp: cfs[hp].unique() for hp in hp_names}
    return hp_dict


def find_basins(grad_f, bounds, smallest_diameter, eps, plateau_eps):
    """Finds all attractive basisn with smallest diameter given. eps tells us when we can stop finding zero i.e. when we have found x s.t. |f(x)| < eps"""
    n = (bounds[1] - bounds[0]) / (smallest_diameter * 2)
    xs = np.linspace(bounds[0], bounds[1], int(n) + 1)
    outs = grad_f(np.array([xs]))[0]

    basins = []
    left_p = bounds[0]
    passed_minimum = False

    for i in range(1, int(n)):
        out = outs[i]

        if (outs[i - 1] >= 0) and (outs[i] <= 0):
            right_p = binary_search_zero(grad_f, np.array([xs[i - 1]]), np.array([xs[i]]), eps)[0]
            basins.append([left_p, right_p])
            left_p = right_p
        elif (abs(outs[i]) < plateau_eps):
            if not passed_minimum:
                passed_minimum = True
            else:
                right_p = xs[i]
                basins.append([left_p, right_p])
                left_p = right_p
                passed_minimum = False

    if left_p != xs[-1]:
        basins.append([left_p, xs[-1]])
    return np.array(basins)


def binary_search_zero(f, a, b, eps):
    while True:
        new_x = (b + a) / 2.
        if abs(f(new_x)) < eps:
            return new_x
        if any(f(new_x) < 0):
            b = new_x
        else:
            a = new_x


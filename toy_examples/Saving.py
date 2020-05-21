import matplotlib.pyplot as plt
import autograd.numpy as np
import time, os, subprocess

from scipy import stats

import pickle
import yaml

import datetime
import socket, sys, os

import plotly.graph_objects as go

from utils import *

# ffmpeg -r 20 -f image2 -s 1920x1080 -i %d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p test.mp4
def create_animation(image_folder, video_path, screen_resolution="1920x1080", framerate=30, qaulity=25,
                     extension=".png"):
    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-r", str(framerate),
            "-f", "image2",
            "-s", screen_resolution,
            "-i", os.path.join(image_folder, "%d" + extension),
            "-vcodec", "libx264",
            "-crf", str(qaulity),
            "-pix_fmt", "yuv420p",
            video_path
        ])


def create_animation_1d_pictures_particles(all_paths, X, Y, ani_path, graph_details={"p_size": 1, "density_function": None}):
    """path: path[:, 0]=path_x, path[:, 1]=path_y"""


    available_colors = ["red", "green"]

    num_tau, num_particles, tau, dim = all_paths.shape

    density_function = graph_details["density_function"]

    for i in range(len(all_paths)):
        curr_paths = all_paths[i]

        color_use = available_colors[i % len(available_colors)]

        for j in range(tau):

            # so we can reuse the axis
            if density_function is not None:
                fig = plt.figure(figsize=(14, 10))
                gs = fig.add_gridspec(4, 1)
                ax = fig.add_subplot(gs[1:4, :])
                ax2 = fig.add_subplot(gs[0, :])
            else:
                fig = plt.figure(figsize=(14, 10))
                ax = fig.add_subplot(1, 1, 1)

            if density_function is not None:
                Y_density = density_function(X, curr_paths[:, j, 0])
                ax2.plot(X, Y_density)

            # fig.suptitle(folder_name, fontsize=20)

            ax.plot(X, Y)
            ax.plot(curr_paths[:, j, 0], curr_paths[:, j, 1], "o", color=color_use, markersize=graph_details["p_size"])

            plt.savefig(os.path.join(ani_path , "{}.png".format(i * tau + j)))

            plt.close()


def create_animation_2d_pictures_particles(all_paths, X, Y, Z, ani_path, graph_details={"p_size": 1, "density_function": None}):
    """path: path[:, 0]=path_x, path[:, 1]=path_y, path[:, 2] = path_z"""

    available_colors = ["red", "green"]

    num_tau, num_particles, tau, dim = all_paths.shape

    density_function = graph_details["density_function"]

    for i in range(len(all_paths)):
        curr_paths = all_paths[i]

        color_use = available_colors[i % len(available_colors)]

        for j in range(tau):

            if density_function is not None:
                fig = plt.figure(figsize=(30, 10))
                gs = fig.add_gridspec(1, 4)
                ax = fig.add_subplot(gs[:, 0:2])
                ax2 = fig.add_subplot(gs[:, 2:4])
            else:
                fig = plt.figure(figsize=(14, 10))
                ax = fig.add_subplot(1, 1, 1)

            if density_function is not None:
                inp = np.array(np.meshgrid(X, Y)).reshape(2, len(X) * len(Y))
                Z_density = density_function(inp, curr_paths[:, j, :2]).reshape(len(X), len(Y))

                if graph_details["type"] == "contour":
                    ax2.contour(X, Y, Z_density, 40)
                else:
                    ax2.imshow(Z_density, cmap=plt.cm.gist_earth_r, extent=[X[0], X[-1], Y[-1], Y[0]],
                              interpolation=graph_details["interpolation"])
            else:
                ax.plot(curr_paths[:, j, 0], curr_paths[:, j, 1], "o", color=color_use, markersize=graph_details["p_size"])

            # fig.suptitle(folder_name, fontsize=20)

            if graph_details["type"] == "contour":
                ax.contour(X, Y, Z, 40)
            else:
                ax.imshow(Z, cmap=plt.cm.gist_earth_r, extent=[X[0], X[-1], Y[-1], Y[0]],
                          interpolation=graph_details["interpolation"])


            plt.savefig(os.path.join(ani_path , "{}.png".format(i * tau + j)))

            plt.close()


# utils
def remove_png(dir_path):
    files = os.listdir(dir_path)
    for item in files:
        if item.endswith(".png"):
            os.remove(os.path.join(dir_path, item))


def save_config(dir_path, config):
    with open(dir_path + "/process.yml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

def get_file_stamp():
    """Return time and hostname as string for saving files related to the current experiment"""
    host_name = socket.gethostname()
    mydate = datetime.datetime.now()
    return "{}_{}".format(mydate.strftime("%b%d_%H-%M-%S"), host_name)

def save_plot(process, path=None):
    # get potential_function and gradient
    U, grad_U = get_potential(process)

    x_low, x_high = process["x_range"]

    if "2d" in process["potential_function"]:

        X = np.linspace(x_low, x_high, 200)
        Y = np.linspace(x_low, x_high, 200)
        inp = np.array(np.meshgrid(X, Y)).reshape(2, len(X) * len(Y))

        out = U(inp)  # grad_U(inp)[0] #
        Z = out.reshape(len(X), len(Y))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_surface(X, Y, Z)

    else:
        X = np.linspace(x_low, x_high, 200)
        inp = np.array([X])

        out = U(inp)

        plt.plot(X, out)

    if path is None:
        plt.show()
    else:
        plt.savefig(os.path.join(path, "plot.png"))







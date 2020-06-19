import matplotlib.pyplot as plt
import time, os, subprocess
import shutil
from  lineages import *


def animation_lineage(sampling_array, val_arr, ani_path, graph_details={"p_size": 1, }):
    T = len(sampling_array)
    curr_lineage, curr_assignments = find_lineages(sampling_array[:2])
    for t in range(2, T):

        # get lineage
        curr_lineage, curr_assignments = find_lineages(sampling_array[t:t + 1],
                                                       starting_lineage=curr_lineage,
                                                       starting_assignments=curr_assignments)
        Ys = get_linages_vals(curr_lineage, val_arr[:t + 1])

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(1, 1, 1)

        for y in Ys:
            ax.plot(list(range(t + 1)), y)

        # fig.suptitle(folder_name, fontsize=20)

        plt.savefig(os.path.join(ani_path, "{}.png".format(t)))

        plt.close()


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


# utils
def remove_png(dir_path):
    files = os.listdir(dir_path)
    for item in files:
        if item.endswith(".png"):
            os.remove(os.path.join(dir_path, item))
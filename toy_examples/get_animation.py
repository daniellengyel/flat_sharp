
import numpy as np
import pandas as pd
import os, pickle, yaml
from postprocessing import *
from analysis import *
from utils import *
from Functions import *

gaussian_type = "simple"

exp_root = "/Users/daniellengyel/flat_sharp/toy_examples/experiments/gaussian/{}/{}"
exp_folder = exp_root.format("{}_gaussian".format(gaussian_type), "low_lr_tmp_beta_tau_sweep")

# exp_root = "/Users/daniellengyel/flat_sharp/toy_examples/experiments/gaussian/{}"
# exp_folder = exp_root.format("May19_12-34-08_Daniels-MacBook-Pro-4.local")



get_animation(exp_folder, "1589822222.868342") #
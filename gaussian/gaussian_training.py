import numpy as np
import matplotlib.pyplot as plt
import copy, yaml
import torch
import torchvision
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from utils import *
from nets.Nets import SimpleNet

from torch.utils.data import DataLoader
from dataloaders import GaussianMixture

import os, time


def train(data, config, folder_path):
    # init torch
    # device = torch.device("cuda:0") # Uncomment this to run on GPU

    torch.backends.cudnn.enabled = False
    if config["torch_random_seed"] is not None:
        torch.manual_seed(config["torch_random_seed"])

    # get data
    train_loader = DataLoader(data[0], batch_size=config["batch_train_size"], shuffle=True)
    test_loader = DataLoader(data[1], batch_size=config["batch_test_size"], shuffle=True)


    # Init neural nets and weights
    num_nets = config["num_nets"]
    net_params = config["net_params"]
    nets = [SimpleNet(*net_params) for _ in range(num_nets)]
    nets_weights = np.zeros(num_nets)

    #  Define a Loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizers = [optim.SGD(nets[i].parameters(), lr=config["learning_rate"],
                            momentum=config["momentum"]) for i in range(num_nets)]

    # Set algorithm params
    weight_type = config["weight_type"]

    beta = config["softmax_beta"]

    # init saving
    file_stamp = time.time() #get_file_stamp()
    writer = SummaryWriter("{}/runs/{}".format(folder_path, file_stamp))
    os.makedirs("{}/models/{}".format(folder_path, file_stamp))
    with open("{}/runs/{}/{}".format(folder_path, file_stamp, "config.yml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # train
    for epoch in range(config["num_epochs"]):
        # get train loaders for each net
        net_data_loaders = [iter(enumerate(train_loader, 0)) for _ in range(num_nets)]

        is_training_epoch = True
        while is_training_epoch:
            # TODO make inner loop a function. That way we can have different methods of looping over data.
            for idx_net in range(num_nets):
                # get net and optimizer
                net = nets[idx_net]
                optimizer = optimizers[idx_net]

                # get the inputs; data is a list of [inputs, labels]
                try:
                    i, data = next(net_data_loaders[idx_net])
                except:
                    is_training_epoch = False
                    break
                inputs, labels = data

                # Compute gradients for input.
                inputs.requires_grad = True

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs.float(), labels)
                loss.backward(retain_graph=True)
                optimizer.step()

                # update weights
                if weight_type == "input_output_forbenius":
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # get input gradients
                    output_forb = torch.norm(outputs)
                    output_forb.backward()
                    input_grads = inputs.grad

                    curr_weight = weight_function_input_jacobian(input_grads)
                    nets_weights[idx_net] += curr_weight
                elif weight_type == "loss_gradient_weights":

                    param_grads = get_grad_params_vec(net)
                    curr_weight = np.linalg.norm(param_grads)
                    nets_weights[idx_net] += curr_weight
                else:
                    raise NotImplementedError()

                # store training accuracy current
                writer.add_scalar('Loss/train/net_{}'.format(idx_net), loss, i + epoch*len(train_loader))

                writer.add_scalar('Potential/curr/net_{}'.format(idx_net), curr_weight, i + epoch*len(train_loader))
                writer.add_scalar('Potential/total/net_{}'.format(idx_net), nets_weights[idx_net], i + epoch*len(train_loader))


            writer.add_scalar('Kish/', kish_effs(nets_weights), i + epoch*len(train_loader))

            # Get variation of network weights
            writer.add_scalar('WeightVarTrace/', np.trace(get_params_var(nets)), i + epoch*len(train_loader))


            # Check resample
            if kish_effs(nets_weights) < config["ess_threshold"]:
                # resample particles
                sampled_idx = sample_index_softmax(nets_weights, nets, beta=beta)
                # init nets etc
                nets = [copy.deepcopy(nets[i]) for i in sampled_idx]
                optimizers = [optim.SGD(nets[i].parameters(), lr=config["learning_rate"],
                                        momentum=config["momentum"]) for i in range(num_nets)]
                nets_weights = np.zeros(num_nets)
                # TODO save which models got swapped how

        # get test error
        for idx_net in range(num_nets):
            correct = 0
            _sum = 0

            for idx, (test_x, test_label) in enumerate(test_loader):
                predict_y = nets[idx_net](test_x.float()).detach()
                predict_ys = np.argmax(predict_y, axis=-1)
                label_np = test_label.numpy()
                _ = predict_ys == test_label
                correct += np.sum(_.numpy(), axis=-1)
                _sum += _.shape[0]

                writer.add_scalar('Accuracy/net_{}'.format(idx_net), correct / _sum, epoch*len(train_loader) + i)

        for idx_net in range(num_nets):
            torch.save(nets[idx_net], '{}/models/{}/net_{}_step_{}.pkl'.format(folder_path, file_stamp, idx_net, epoch*len(train_loader) + i))



    return nets




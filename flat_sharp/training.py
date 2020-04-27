import numpy as np
import matplotlib.pyplot as plt
import copy, yaml, pickle
import torch
import torchvision
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from utils import *
from nets.Nets import SimpleNet, LeNet

from torch.utils.data import DataLoader

import os, time


def train(config, folder_path, train_data, test_data):
    # init torch
    # device = torch.device("cuda:0") # Uncomment this to run on GPU

    torch.backends.cudnn.enabled = False
    if config["torch_random_seed"] is not None:
        torch.manual_seed(config["torch_random_seed"])
        np.random.seed(config["torch_random_seed"])

    # get dataloaders
    train_loader = DataLoader(train_data, batch_size=config["batch_train_size"], shuffle=True)
    test_loader = DataLoader(test_data, batch_size=config["batch_test_size"], shuffle=True)


    # Init neural nets and weights
    num_nets = config["num_nets"]
    if config["net_name"] == "SimpleNet":
        nets = [SimpleNet(*config["net_params"]) for _ in range(num_nets)]
    elif config["net_name"] == "LeNet":
        nets = [LeNet(*config["net_params"]) for _ in range(num_nets)]
    else:
        raise NotImplementedError("{} is not implemented.".format(config["net_name"]))
    nets_weights = np.zeros(num_nets)

    #  Define a Loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizers = [optim.SGD(nets[i].parameters(), lr=config["learning_rate"],
                            momentum=config["momentum"]) for i in range(num_nets)]

    # Set algorithm params
    weight_type = config["weight_type"]
    beta = config["softmax_beta"]
    ess_threshold = config["ess_threshold"]
    sampling_tau = config["sampling_tau"]
    assert not ((ess_threshold is not None) and (sampling_tau is not None))
    if ess_threshold is not None:
        should_resample = lambda w, s: (kish_effs(w) < ess_threshold) and (beta != 0)
    elif sampling_tau is not None:
        should_resample = lambda w, s: (s % sampling_tau == 0) and (s > 0)
    else:
        should_resample = lambda w, s: False

    num_steps = config["num_steps"]
    mean_loss_threshold = config["mean_loss_threshold"]
    assert not ((num_steps is not None) and (mean_loss_threshold is not None))
    if num_steps is not None:
        stopping_criterion = lambda ml, s: num_steps < s
    elif mean_loss_threshold is not None:
        stopping_criterion = lambda ml, s: ml < mean_loss_threshold
    else:
        raise Exception("Error: Did not provide a stopping criterion.")

    # init saving
    file_stamp = str(time.time())  # get_file_stamp()
    writer = SummaryWriter("{}/runs/{}".format(folder_path, file_stamp))
    with open("{}/runs/{}/{}".format(folder_path, file_stamp, "config.yml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # save init nets
    os.makedirs(os.path.join(folder_path, "resampling", file_stamp, "step_{}".format(0), "models"))

    for idx_net in range(num_nets):
        torch.save(nets[idx_net],
                   os.path.join(folder_path, "resampling", file_stamp, "step_{}".format(0), "models",
                                "net_{}.pkl".format(idx_net)))
    sampled_idx = np.array(list(range(0, num_nets)))
    with open(os.path.join(folder_path, "resampling", file_stamp, "step_{}".format(0), "sampled_idx.pkl"),
              "wb") as f:
        pickle.dump(sampled_idx, f)

    # init number of steps taken to first step
    curr_step = 1
    mean_loss = float("inf")

    # train
    while (not stopping_criterion(mean_loss, curr_step)):

        # get train loaders for each net
        net_data_loaders = [iter(train_loader) for _ in range(num_nets)]

        is_training_curr = True
        while is_training_curr and (not stopping_criterion(mean_loss, curr_step)):
            if (curr_step % 100) == 1:
                print("Step: {}".format(curr_step))
                print("Mean Loss: {}".format(mean_loss))

            # do update step for each net
            nets, nets_weights, took_step, mean_loss_after_step = _training_step(nets, nets_weights, optimizers, net_data_loaders, criterion, weight_type, curr_step=curr_step, writer=writer)

            # if is_training_curr is returned false by _training_step it means we didn't take a step
            if not took_step:
                break
            else:
                mean_loss = mean_loss_after_step

            # store global metrics
            writer.add_scalar('Kish/', kish_effs(nets_weights), curr_step)
            # # Get variation of network weights
            covs = get_params_cov(nets)
            writer.add_scalar('WeightVarTrace/', torch.norm(covs), curr_step)

            # Check resample
            if should_resample(nets_weights, curr_step):
                # save nets
                os.makedirs(os.path.join(folder_path, "resampling", file_stamp, "step_{}".format(curr_step), "models"))

                # for idx_net in range(num_nets):
                #     torch.save(nets[idx_net],
                #                os.path.join(folder_path, "resampling", file_stamp, "step_{}".format(curr_step),
                #                             "models", "net_{}.pkl".format(idx_net)))

                # resample particles
                if beta != 0:
                    sampled_idx = sample_index_softmax(nets_weights, nets, beta=beta)
                    nets = [copy.deepcopy(nets[i]) for i in sampled_idx]
                    optimizers = [optim.SGD(nets[i].parameters(), lr=config["learning_rate"],
                                            momentum=config["momentum"]) for i in range(num_nets)]
                else:
                    sampled_idx = list(range(num_nets))


                nets_weights = np.zeros(num_nets)

                # save the resample indecies
                with open(os.path.join(folder_path, "resampling", file_stamp, "step_{}".format(curr_step), "sampled_idx.pkl"), "wb") as f:
                    pickle.dump(sampled_idx, f)

                is_training_curr = False

            # update curr_step
            curr_step += 1

        # get test error
        for idx_net in range(num_nets):
            accuracy = get_net_accuracy(nets[idx_net], test_loader)
            writer.add_scalar('Accuracy/net_{}'.format(idx_net), accuracy, curr_step)

    # save final nets
    os.makedirs(os.path.join(folder_path, "resampling", file_stamp, "step_{}".format(curr_step), "models"))

    for idx_net in range(num_nets):
        torch.save(nets[idx_net],
                   os.path.join(folder_path, "resampling", file_stamp, "step_{}".format(curr_step), "models",
                                "net_{}.pkl".format(idx_net)))
    sampled_idx = np.array(list(range(0, num_nets)))
    with open(os.path.join(folder_path, "resampling", file_stamp, "step_{}".format(curr_step), "sampled_idx.pkl"),
              "wb") as f:
        pickle.dump(sampled_idx, f)
    return nets


def _training_step(nets, nets_weights, net_optimizers, net_data_loaders, criterion, weight_type, var_noise=None, curr_step=0, writer=None):
    """Does update step on all networks and computes the weights.
    If wanting to do a random walk, set learning rate of net_optimizer to zero and set var_noise to noise level."""
    taking_step = True

    mean_loss = 0

    for idx_net in range(len(nets)):

        # get net and optimizer
        net = nets[idx_net]
        optimizer = net_optimizers[idx_net]

        # get the inputs; data is a list of [inputs, labels]
        try:
            data = next(net_data_loaders[idx_net])
        except:
            taking_step = False
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

        if var_noise is not None:
            with torch.no_grad():
                for param in net.parameters():
                    param.add_(torch.randn(param.size()) * var_noise)

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
            curr_weight = torch.norm(param_grads)
            nets_weights[idx_net] += curr_weight
        elif weight_type is None:
            pass
        else:
            raise NotImplementedError()

        # store metrics for each net
        if writer is not None:
            writer.add_scalar('Loss/train/net_{}'.format(idx_net), loss, curr_step)
            writer.add_scalar('Potential/curr/net_{}'.format(idx_net), curr_weight, curr_step)
            writer.add_scalar('Potential/total/net_{}'.format(idx_net), nets_weights[idx_net], curr_step)
            writer.add_scalar('Norm/net_{}'.format(idx_net), torch.norm(get_params_vec(net)), curr_step)

        mean_loss += float(loss)

    assert taking_step or (idx_net == 0)

    return nets, nets_weights, taking_step, mean_loss/len(nets)


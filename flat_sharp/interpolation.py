import numpy as np
from utils import *
from data_getters import *
from postprocessing import *

import copy
import torch


def interpolate_models(model1, model2, beta):
    params1 = model1.named_parameters()
    params2 = model2.named_parameters()

    new_model = copy.deepcopy(model2)
    new_params = new_model.named_parameters()
    dict_new_params = dict(new_params)
    for name1, param1 in params1:
        if name1 in dict_new_params:
            dict_new_params[name1].data.copy_((1. - beta) * param1.data + beta * dict_new_params[name1].data)

    return new_model

def scale_output_model(model1, alpha):
    if isinstance(model1, LeNet):
        last_layer_names = ["fc3.weight", "fc3.bias"]
    else:
        last_layer_names = ["fc2.weight", "fc2.bias"]

    params1 = model1.named_parameters()

    new_model = copy.deepcopy(model1)
    new_params = new_model.named_parameters()
    dict_new_params = dict(new_params)
    for name1, param1 in params1:
        if name1 in last_layer_names:
            dict_new_params[name1].data.copy_(alpha * param1.data)
            
    return new_model


def T_alpha_models(model, num_inter_models, alpha_range):
    inter_models_arr = []

    alphas = np.linspace(alpha_range[0], alpha_range[1], num_inter_models)
    for alpha in alphas:

        params1 = model.named_parameters()

        new_model = copy.deepcopy(model)
        new_params = new_model.named_parameters()
        dict_new_params = dict(new_params)
        for name1, param1 in params1:
            if name1 in dict_new_params:
                dict_new_params[name1].data.copy_((1. - beta) * param1.data + beta * dict_new_params[name1].data)


        inter_models_arr.append(curr_model)


    return inter_models_arr



    return new_model


def get_loss_grad(net, criterion, data):
    inputs, labels = data

    # Compute gradients for input.
    inputs.requires_grad = True

    net.zero_grad()

    # forward + backward + optimize
    outputs = net(inputs)
    loss = criterion(outputs.float(), labels)
    loss.backward(retain_graph=True)

    param_grads = get_grad_params_vec(net)
    return loss, torch.norm(param_grads)


def get_model_interpolate_arr(model_a, model_b, num_inter_models, beta_bound=None):
    inter_models_arr = []
    if beta_bound is None:
        beta_bound = [0, 1]
    betas = np.linspace(beta_bound[0], beta_bound[1], num_inter_models)
    for beta in betas:
        curr_model = interpolate_models(model_a, model_b, beta)

        inter_models_arr.append(curr_model)

    return inter_models_arr


def get_model_interpolate_2d(offset, v1, v2, num_inter_models, alpha_bound, beta_bound, func):
    X = np.linspace(alpha_bound[0], alpha_bound[1], num_inter_models)
    Y = np.linspace(beta_bound[0], beta_bound[1], num_inter_models)

    v1_net = vec_to_net(v1, offset)
    v2_net = vec_to_net(v2, offset)

    v1_dict = dict(v1_net.named_parameters())
    v2_dict = dict(v2_net.named_parameters())

    val_arr = []


    for x in X:
        curr_arr = []

        for y in Y:

            curr_model = copy.deepcopy(offset)
            dict_curr_model = dict(curr_model.named_parameters())

            for name1, param1 in offset.named_parameters():
                dict_curr_model[name1].data.copy_(dict_curr_model[name1].data + x * v1_dict[name1].data + y * v2_dict[name1].data)

            to_append = func(curr_model)
            curr_arr.append(to_append)

        val_arr.append(curr_arr)

    return val_arr


def project_onto(net, v1, v2, offset):
    v1_norm = v1 / torch.norm(v1)
    v2_norm = v2 / torch.norm(v2)

    net_vect = get_params_vec(net) - get_params_vec(offset)
    alpha = torch.matmul(v1_norm, net_vect)
    beta = torch.matmul(v2_norm, net_vect)

    return alpha, beta

def take_n_gd_steps(net, optimizer, criterion, data, n=1, get_grad=True, v1=None, v2=None, offset=None):
    grads_arr = []

    projections = []

    if (v1 is not None) and (v2 is not None):
        projections.append(project_onto(net, v1, v2, offset))

    for _ in range(n):
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

        if (_ % 100) == 0:
            print(_)
            print(loss)
            print()

        if get_grad:
            grads_arr.append(get_grad_params_vec(net))

        if (v1 is not None) and (v2 is not None):
            projections.append(project_onto(net, v1, v2, offset))

    return net, grads_arr, projections


def do_the_do(model, optimizer, criterion, data_loader, num_inter_models, num_steps=1, beta_bound=None):
    data = next(iter(data_loader))

    model_a = copy.deepcopy(model)
    model_b = take_n_gd_steps(model, optimizer, criterion, data, n=num_steps)

    inter_models = get_model_interpolate_arr(model_a, model_b, num_inter_models, beta_bound=beta_bound)
    return inter_models

exp_id = "1589992134.56161"

if __name__ == "__main__":
    # get data
    train_data, test_data = get_postprocessing_data(experiment_folder, vectorized=True)
    train_loader = DataLoader(train_data, batch_size=10000, shuffle=True)  # fix the batch size
    test_loader = DataLoader(test_data, batch_size=len(test_data))

    criterion = torch.nn.CrossEntropyLoss()
    cfs_dict = exp_dict["stuff"]["configs"].loc[exp_id].to_dict()
    nets = get_nets(cfs_dict)
    optimizers = get_optimizers(cfs_dict)(nets)
    inter_nets = []
    for nn_idx in range(len(nets)):
        inter_nets.append(do_the_do(nets[nn_idx], optimizers[nn_idx], criterion, train_loader, 20))


    for nn_index in range(len(nets)):
        y_val = inter_nets[nn_index][1][:, 1]
        plt.plot(list(range(len(y_val))), y_val)
        plt.show()

def interpolate_models(model1, model2, beta):
    params1 = model1.named_parameters()
    params2 = model2.named_parameters()

    dict_params2 = dict(params2)

    for name1, param1 in params1:
        if name1 in dict_params2:
            dict_params2[name1].data.copy_((1 - beta) * param1.data + beta * dict_params2[name1].data)

    return model2


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


def get_model_interpolate_arr(model_a, model_b, criterion, data, num_inter_models):
    inter_models_arr = []
    inter_loss_grad_arr = []

    betas = np.linspace(0, 1, num_inter_models)
    for beta in betas:
        curr_model = copy.deepcopy(model_b)
        curr_model = interpolate_models(model_a, curr_model, beta)
        curr_loss, curr_grad = get_loss_grad(curr_model, criterion, data)

        inter_models_arr.append(curr_model)
        inter_loss_grad_arr.append([curr_loss, curr_grad])

    return inter_models_arr, np.array(inter_loss_grad_arr)


def _take_n_gd_steps(net, optimizer, criterion, data, n=1):
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

    return net


def do_the_do(model, optimizer, criterion, data_loader, num_inter_models):
    data = next(iter(data_loader))

    model_a = copy.deepcopy(model)
    model_b = _take_n_gd_steps(model, optimizer, criterion, data)

    return get_model_interpolate_arr(model_a, model_b, criterion, data, num_inter_models)


exp_id = "1589992134.56161"

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
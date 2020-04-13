import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import os

PATH_TO_DATA = "/Users/daniellengyel/flat_sharp/flat_sharp/data"

def get_data(data_name):
    if data_name == "gaussian":
        return _get_gaussian()
    elif data_name == "MNIST":
        return _get_MNIST()
    else:
        raise NotImplementedError("{} is not implemented.".format(data_name))

def _get_MNIST():
    train_data = torchvision.datasets.MNIST(os.path.join(PATH_TO_DATA, "MNIST"), train=True,
                                            download=False,
                                            transform=torchvision.transforms.Compose([
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(
                                                    (0.1307,), (0.3081,))
                                            ]))
    test_data = torchvision.datasets.MNIST(os.path.join(PATH_TO_DATA, "MNIST"), train=False,
                                           download=False,
                                           transform=torchvision.transforms.Compose([
                                               torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize(
                                                   (0.1307,), (0.3081,))
                                           ]))

    train_data, test_data = MNISTWrapper(train_data), MNISTWrapper(test_data)

    return train_data, test_data

def _get_gaussian():
    # get data
    gaussian_params = []

    cov_1 = np.array([[1, 1 / 2.], [1 / 2., 1]])
    cov_2 = np.array([[1, 1 / 2.], [1 / 2., 1]])

    mean_1 = np.array([0, 0])
    mean_2 = np.array([2, 0])

    means = [mean_1, mean_2]
    covs = [cov_1, cov_2]
    training_nums = 500
    test_nums = 100

    train_gaussian = GaussianMixture(means, covs, len(means) * [training_nums])
    test_gaussian = GaussianMixture(means, covs, len(means) * [test_nums])

    return train_gaussian, test_gaussian

#  specific for current data
class MNISTWrapper():
    def __init__(self, MNIST_Dataset):
        self.MNIST = MNIST_Dataset

    def __getitem__(self, item):
        data, target = self.MNIST.__getitem__(item)
        return data.view(-1), target

    def __len__(self):
        return len(self.MNIST)


class GaussianMixture(Dataset):
    """Dataset gaussian mixture. Points of first gaussian are mapped to 0 while points in the second are mapped 1.

    Parameters
    ----------
    means:
        i: mean
    covs:
        i: cov
    nums:
        i: num for ith class
    """

    def __init__(self, means, covs, nums):
        self.data = []
        self.targets = []

        self.num_classes = len(covs)

        xs = None
        ys = []
        for i in range(len(covs)):
            mean = means[i]
            cov = covs[i]
            num = nums[i]
            x = np.random.multivariate_normal(mean, cov, num)
            if xs is None:
                xs = x
            else:
                xs = np.concatenate([xs, x], axis=0)
            ys += num * [i]

        self.data = torch.Tensor(xs)

        targets = np.array(ys)  # np.eye(self.num_classes)[ys]
        self.targets = torch.Tensor(targets)

    def __getitem__(self, index):
        return self.data[index], self.targets[index].long()

    def __len__(self):
        return len(self.data)


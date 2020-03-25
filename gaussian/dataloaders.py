import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


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

        targets = np.array(ys) #np.eye(self.num_classes)[ys]
        self.targets = torch.Tensor(targets)

    def __getitem__(self, index):
        return self.data[index], self.targets[index].long()

    def __len__(self):
        return len(self.data)

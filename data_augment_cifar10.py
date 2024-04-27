"""
misc file for augmenting cifar10 using routines from timm

"""
import time
import functools
import torch.nn as nn
import torch as t
import numpy as np
import torchvision

import torchvision.transforms as tt
from torchvision.transforms.functional import to_pil_image, pil_to_tensor
from timm.data.auto_augment import rand_augment_transform
from PIL import Image as im
from matplotlib import pyplot as plt

def enlarge_cifar10_dataset(data_x, data_y, n_enlarge=20):
    n_data = data_x.shape[0]

    total_data_x = []
    total_data_y = []

    image_means = [int(data_x[:, :, :, 0].mean()), int(data_x[:, :, :, 1].mean()), int(data_x[:, :, :, 2].mean())]

    tfm = rand_augment_transform(
        config_str='rand-m9-n3-mstd1',
        hparams={'translate_const': 10, 'img_mean': tuple(image_means)}
    )

    for i in range(n_data):

        if i%100 == 0:
            print(f"finished augmenting im {i}/{n_data}")

        x = im.fromarray(data_x[i])
        y = data_y[i]
        for j in range(n_enlarge):
            total_data_x.append(np.array(tfm(x)))
            total_data_y.append(y)

    return np.stack(total_data_x), np.array(total_data_y)


if __name__ == "__main__":

    import os

    os.makedirs('./datasets', exist_ok=True)

    # ========================================================================================
    # Loading datasets and augmenting them
    # ========================================================================================

    data_train = torchvision.datasets.CIFAR10('./datasets/', train=True, download=True)
    data_test = torchvision.datasets.CIFAR10('./datasets/', train=False, download=True)

    data_x = data_train.data
    data_y = data_train.targets

    augmented_x, augmented_y = enlarge_cifar10_dataset(data_x, data_y, n_enlarge=20)

    # ========================================================================================
    # Normalizing datasets to mean zero and unit variance
    # ========================================================================================

    data_x = t.from_numpy(data_x).float().transpose(1, 3)
    data_y = t.LongTensor(data_y)

    test_data_x = t.from_numpy(data_test.data).float().transpose(1, 3)
    test_data_y = t.LongTensor(data_test.targets)

    augmented_x = t.from_numpy(augmented_x).float().transpose(1, 3)
    augmented_y = t.LongTensor(augmented_y)

    x_mean = data_x.mean(dim=0)
    x_std = data_x.std(dim=0)

    x_mean_aug = augmented_x.mean(dim=0)
    x_std_aug = augmented_x.std(dim=0)

    data_x = (data_x - x_mean) / (1e-7 + x_std)
    augmented_x = (augmented_x - x_mean_aug) / (1e-7 + x_std_aug)

    test_data_x_not_aug = (test_data_x - x_mean) / (1e-7 + x_std)
    test_data_x_aug = (test_data_x - x_mean_aug) / (1e-7 + x_std_aug)

    # ========================================================================================
    # saving datasets for ease of use with rest of code
    # ========================================================================================

    t.save((data_x, data_y), './datasets/cifar10_training.pth')
    t.save((augmented_x, augmented_y), './datasets/cifar10_training_augmented.pth')

    t.save((test_data_x_not_aug, test_data_y), './datasets/cifar10_test_not_augmented.pth')
    t.save((test_data_x_aug, test_data_y), './datasets/cifar10_test_augmented.pth')


import numpy as np
import torch as t
from sklearn.manifold import Isomap
from matplotlib import pyplot as plt

def generate_figures():
    cifar10_training = t.load('./datasets/cifar10_training.pth')
    cifar10_training_augmented = t.load('./datasets/cifar10_training_augmented.pth')

    # cifar10_test_not_augmented = t.load('./datasets/cifar10_test_not_augmented.pth')
    # cifar10_test_augmented = t.load('./datasets/cifar10_test_augmented.pth')

    sets = [
        ("cifar10_training", "pas augmenté", *cifar10_training),
        ("cifar10_training_augmented", "augmenté", *cifar10_training_augmented),
    ]

    for dataset, variant, x, y in sets:
        shape = x.shape
        reshaped = x.reshape(shape[0], np.prod(shape[1:]))
        N_samples = 4000
        samples = np.random.choice(shape[0], N_samples)

        subsampled_images = reshaped[samples, :]
        subsampled_labels = y[samples]

        isomap = Isomap(n_components=3)
        new_coords = isomap.fit_transform(subsampled_images)

        print(new_coords.shape)
        # On fabrique une grille
        # x0, x1 = min(new_coords[:, 0]), max(new_coords[:, 0])
        # y0, y1 = min(new_coords[:, 1]), max(new_coords[:, 1])
        # nxn = 10
        # xsize = (x1 - x0)/nxn
        # ysize = (y1 - y0)/nxn
        # xx, yy = np.meshgrid(np.linspace(x0, x1, nxn), np.linspace(y0, y1, nxn))

        # On crée la figure
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.scatter(new_coords[:, 0], new_coords[:, 1], c=subsampled_labels, alpha=0.5)
        #plt.title(f"Plongement 2D des images par isomap ({variant})")
        fig.savefig(f"{dataset}_2d.png", dpi=500)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(new_coords[:, 0], new_coords[:, 1], new_coords[:, 2], c=subsampled_labels, alpha=0.5)
        #plt.title(f"Plongement 3D des images par isomap ({variant})")
        fig.savefig(f"{dataset}_3d.png", dpi=500)

def test():
    x, y = t.load('./datasets/cifar10_training.pth')

    shape = x.shape
    reshaped = x.reshape(shape[0], np.prod(shape[1:]))
    N_samples = 4000
    samples = np.random.choice(shape[0], N_samples)

    subsampled_images = reshaped[samples, :]
    subsampled_labels = y[samples]

    isomap = Isomap(n_components=3)
    new_coords = isomap.fit_transform(subsampled_images)
    return new_coords

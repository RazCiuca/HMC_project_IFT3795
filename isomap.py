import numpy as np
import torch as t
from sklearn.manifold import Isomap
from matplotlib import pyplot as plt

def generate_figures():
    cifar10_training = t.load('./datasets/cifar10_training.pth')
    cifar10_training_augmented = t.load('./datasets/cifar10_training_augmented.pth')

    # cifar10_test_not_augmented = t.load('./datasets/cifar10_test_not_augmented.pth')
    # cifar10_test_augmented = t.load('./datasets/cifar10_test_augmented.pth')

    things = [
        ("cifar10_training", *cifar10_training),
        ("cifar10_training_augmented", *cifar10_training_augmented),
    ]

    for title, x, y in things:
        shape = x.shape
        reshaped = x.reshape(shape[0], np.prod(shape[1:]))
        N_samples = 4000
        samples = np.random.choice(shape[0], N_samples)

        subsampled_images = reshaped[samples, :]
        subsampled_labels = y[samples]

        isomap = Isomap()
        new_coords = isomap.fit_transform(subsampled_images)

        # On fabrique une grille
        x0, x1 = min(new_coords[:, 0]), max(new_coords[:, 0])
        y0, y1 = min(new_coords[:, 1]), max(new_coords[:, 1])
        nxn = 10
        xsize = (x1 - x0)/nxn
        ysize = (y1 - y0)/nxn
        xx, yy = np.meshgrid(np.linspace(x0, x1, nxn), np.linspace(y0, y1, nxn))

        # On crée la figure
        plt.scatter(new_coords[:, 0], new_coords[:, 1], c=subsampled_labels, alpha=0.5)
        plt.title("Plongement 2D des images par isomap")
        plt.savefig(f"{title}.png", dpi=500)
        plt.clf()

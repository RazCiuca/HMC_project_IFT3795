import data_augment_cifar10
import torchvision
from PIL import Image

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def augmented():
    testset = torchvision.datasets.CIFAR10('./datasets/', train=False, download=True)
    images = testset.data
    targets = testset.targets

    N = 1000

    (xs, ys) = data_augment_cifar10.enlarge_cifar10_dataset(images[0:N,:,:,:], targets[0:N], n_enlarge=1)

    images = []
    for i in range(N):
        image = Image.fromarray(xs[i,:,:,:])
        images.append((image, ys[i]))

    return images

def savefig():
    aug = [v for (v, i) in augmented()]
    im = image_grid(aug[:100], 10, 10)
    im.save("fig.png", "PNG")

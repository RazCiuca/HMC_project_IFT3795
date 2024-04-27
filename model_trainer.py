
import torch as t
import torch.nn as nn
import numpy as np
import torch.optim as optim

def train(model, data_x, data_y, loss_fn, device, batch_size=256,
          lr=1e-3, n_iter=20000, weight_decay=1e-2, verbose=True, training_run_name=''):

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    n_data = data_x.size(0)

    # training loop

    # indices that we loop through
    all_indices = np.random.permutation(np.arange(n_data))
    all_indices = np.concatenate([all_indices, all_indices[:batch_size].copy()])
    model.train()

    model = model.to(device)

    for iter in range(n_iter):

        # sample batch, send to gpu
        i1 = (iter * batch_size) % n_data
        i2 = i1 + batch_size
        batch_indices = all_indices[i1:i2]

        # batch_indices_total.append(t.from_numpy(batch_indices))

        inputs = data_x[batch_indices].to(device)
        targets = data_y[batch_indices].to(device)

        # pass through model
        preds = model(inputs)

        # compute loss and append to list
        loss = loss_fn(preds, targets)

        if verbose:
            if iter % 100 == 0 and iter != 0:
                print(training_run_name + f" iter {iter}, loss:{loss.item():.5f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model

if __name__ == "__main__":

    from resnet import *
    from polynomial_models import *

    # example usage:

    data_x, data_y = t.load('./datasets/cifar10_training_augmented.pth')

    model = ResNet9(3, 10, expand_factor=2)

    loss_fn = nn.CrossEntropyLoss()

    device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')

    trained_model = train(model, data_x, data_y, loss_fn, device, batch_size=256,
                    lr=1e-3, n_iter=20000, weight_decay=1e-2, verbose=True, training_run_name='test_run')



import torch as t
import torch.nn as nn
import numpy as np
import torch.optim as optim


def get_validation(model, data_x, targets):

    with t.no_grad():
        model.eval()

        top_k_accuracy = []

        chunk_size = 128
        n_data = data_x.size(0)

        preds = t.cat([model(data_x[i*chunk_size:(i+1)*chunk_size]) for i in range(int(n_data/chunk_size)+1) ], dim=0)

        sorted_preds = t.argsort(preds, dim=1)

        correct_preds = sorted_preds[:, -1] == targets

        accuracy = t.mean((correct_preds).float())

        return accuracy


def train(model, data_x, data_y, loss_fn, device, batch_size=256,
          lr=1e-3, momentum=0.9, n_iter=20000, weight_decay=1e-2, verbose=True, training_run_name='',
          validation_dataset=None):

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    n_data = data_x.size(0)

    # training loop

    # indices that we loop through
    all_indices = np.random.permutation(np.arange(n_data))
    all_indices = np.concatenate([all_indices, all_indices[:batch_size].copy()])

    if validation_dataset is not None:
        data_x_val, data_y_val = validation_dataset

    model = model.to(device)

    for iter in range(n_iter):
        model.train()

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
            if iter % 200 == 0 and iter != 0:

                val_accuracy = 0.0

                if validation_dataset is not None:
                    val_accuracy = get_validation(model, data_x_val, data_y_val)

                print(training_run_name + f" iter {iter}, loss:{loss.item():.5f}, acc: {val_accuracy:.4f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model

if __name__ == "__main__":

    from resnet import *
    from polynomial_models import *
    device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')

    # ================================================================================
    # Defining Models and Loading Datasets
    # ================================================================================

    data_x, data_y = t.load('./datasets/cifar10_training.pth')
    # data_x, data_y = t.load('./datasets/cifar10_training_augmented.pth')
    validation_dataset = (x.to(device) for x in t.load('./datasets/cifar10_test_augmented.pth'))
    n_data = data_x.size(0)

    model = ResNet9(3, 10, expand_factor=2)
    # model = PolynomialRegressor(3*16*16, n_out=10, poly_degree=2, avg_pool_size=2)
    # model = PolynomialRegressor(3*8*8, n_out=10, poly_degree=2, avg_pool_size=4)
    # model = PolynomialRegressor(3*4*4, n_out=10, poly_degree=3, avg_pool_size=8)

    # model = PolynomialRegressor(3*32*32, n_out=10, poly_degree=1, avg_pool_size=None)

    # ================================================================================
    # Training The model
    # ================================================================================

    loss_fn = nn.CrossEntropyLoss()

    trained_model = train(model, data_x, data_y, loss_fn, device, batch_size=512,
                    lr=1e-2, momentum=0.98, n_iter=10000, weight_decay=0.0, verbose=True, training_run_name='test_run',
                          validation_dataset=validation_dataset)

    # ================================================================================
    # Computing Top eigenvalues
    # ================================================================================

    from HessianEigen_utils import *
    from nuts import nuts6, eigen_r0_sampler, sampling_from_eigen
    from BayesAveragedModel import *

    eigvals, eigvecs = top_k_hessian_eigen(trained_model, data_x, data_y, loss_fn, top_k=100, mode='LA', batch_size=5000)

    print('top eigenvalues are: ')
    print(eigvals)

    eigvals = eigvals * n_data
    eigvecs = eigvecs.T

    def r0_sampler():
        return eigen_r0_sampler(eigvals, eigvecs)


    # ================================================================================
    # sampling purely from the eigenvalues we know
    # ================================================================================
    initial_params_pt = trained_model.get_vectorized_params().unsqueeze(0)

    param_samples = t.from_numpy(sampling_from_eigen(eigvals, eigvecs, 5000)).to(device).float() + initial_params_pt

    param_samples = param_samples.to(device)

    averaged_model = BayesAveragedModel(trained_model, param_samples)
    data_x_val, targets_val = (x.to(device) for x in t.load('./datasets/cifar10_test_augmented.pth'))
    averaged_val = get_validation(averaged_model, data_x_val, targets_val)

    print(f"averaged validation for easy eigensampling: {averaged_val}")

    # ================================================================================
    # Using Nuts to Sample from the posterior
    # ================================================================================

    initial_params_pt = trained_model.get_vectorized_params()
    initial_params = initial_params_pt.clone().detach().cpu().numpy()


    def log_loss_grad(theta):
        # load theta into model
        theta = t.from_numpy(theta).to(device).float()

        for p1, p2 in zip(trained_model.parameters(), trained_model.shape_vec_as_params_no_names(theta)):
            p1.data = p2

        indices = np.random.permutation(np.arange(data_x.size(0)))[:2048]
        data_x_p = data_x[indices]
        data_y_p = data_y[indices]

        loss, gradients = grad_model(trained_model, data_x_p, data_y_p, loss_fn, chunk_size=2048)

        return n_data*loss, n_data*gradients.cpu().numpy()

    samples, lnprob, epsilon = nuts6(log_loss_grad, M=1000, Madapt=1000, theta0=initial_params, r0_sampler=r0_sampler)

    samples_pt = t.from_numpy(samples).float().to(device)
    averaged_model = BayesAveragedModel(trained_model, samples_pt)
    data_x_val, targets_val = (x.to(device) for x in t.load('./datasets/cifar10_test_augmented.pth'))
    averaged_val = get_validation(averaged_model, data_x_val, targets_val)

    # print(samples.shape)
    print(f"averaged val:{averaged_val}")


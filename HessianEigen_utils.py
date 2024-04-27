"""
This file defines routines for computing the entire hessian of a model row-by-row, sending things
back to cpu, and storing the result on disk

steps for gaussian computation:

for each row, compute the Hessian-vector product on gpu, then send to cpu. If the dataset is too big, split it
up into chunks and sum the Hv product over the chunks. Periodically store the pieces to disk, in case it crashes,
and start back the process by reading them from dist. Then load up the pieces and concatenate them.

"""

import time
import torch as t
import numpy as np
import torch.optim as optim
import torchvision
from torch.func import functional_call
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from scipy.sparse.linalg import LinearOperator, eigsh

from torch.func import jvp, grad, vjp
from torch.autograd.functional import vhp

def hvp_func(f, primals, tangents):
  return jvp(grad(f), primals, tangents)[1:]

def hvp_model_np(tangents_np, model, data_x, data_y, loss_fn):
    tangents = t.from_numpy(tangents_np).to('cuda').squeeze()
    tangents = tuple(model.shape_vec_as_params_no_names(tangents))

    # send data to gpu
    data_x = data_x.cuda()
    data_y = data_y.cuda()

    # defining the function to send to hvp
    def fn_to_optim(*x):
        z = [{name: p} for (p, name) in zip(x, model.param_names)]

        preds = functional_call(model, z, data_x)
        return loss_fn(preds, data_y)

    hessian_col = vhp(fn_to_optim, tuple(model.parameters()), tangents)[1]

    return t.cat([x.flatten() for x in hessian_col]).detach().cpu().numpy()

def grad_model(model, data_x, data_y, loss_fn, chunk_size=128):

    # send data to gpu
    data_x = data_x.cuda()
    data_y = data_y.cuda()

    n_data = data_x.size(0)
    n_iter = int(n_data/chunk_size) + 1

    gradients = t.zeros(model.get_vectorized_params().size())

    total_loss = 0

    for i in range(n_iter):
        i1 = i*chunk_size
        i2 = (i+1)*chunk_size

        data_x = data_x[i1:i2]
        data_y = data_y[i1:i2]

        preds = model(data_x)

        loss = loss_fn(preds, data_y)

        grads = t.autograd.grad(loss, model.parameters())
        gradients += (i2-i1) * t.cat([x.flatten() for x in grads]).detach().cpu()

        total_loss += loss.item()*(i2-i1)

    return total_loss/n_data, gradients/n_data

def top_k_hessian_eigen(model, data_x, data_y, loss_fn, top_k = 100, mode='LA', batch_size=None):
    """
    computes top-k eigenvalues and eigenvectors of the hessian of model, with given data,
    possibly with finite batch size
    """
    # if finite batch size, resample this at every computation?
    if batch_size is not None:
        indices = t.randperm(data_x.size(0), device=data_x.device)[:batch_size]
        data_x = data_x[indices]
        data_y = data_y[indices]


    linop = LinearOperator((model.n_params, model.n_params),
                            matvec = lambda x: hvp_model_np(x, model, data_x, data_y, loss_fn))

    eigvals, eigvecs = eigsh(linop, k=top_k, which=mode)

    n_samples = data_x.size(0)

    return eigvals, eigvecs

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

def grad_model(model, data_x, data_y, loss_fn):

    # send data to gpu
    data_x = data_x.cuda()
    data_y = data_y.cuda()

    preds = model(data_x)

    loss = loss_fn(preds, data_y)

    gradients = t.autograd.grad(loss, model.parameters())

    return t.cat([x.flatten() for x in gradients]).detach().cpu()

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


class HessianCompute:

    def __init__(self, dataset, model, loss_fn):
        super(HessianCompute, self).__init__()

        self.data_x, self.data_y = dataset
        self.loss_fn = loss_fn
        self.model = model
        self.params = tuple(model.parameters())
        self.vec_params = self.model.get_vectorized_params()
        self.n_params = self.vec_params.size(0)

        self.batch_chunksize = 10000
        self.n_chunks = int(self.data_x.size(0)/self.batch_chunksize)

        self.chunk_size_to_save = 10**9

    def compute_hessian(self):
        """
        iterate over the dataset, computing hessian vector products, then sending them to gpu
        :return:
        """
        total_hess = []

        # for loop over columns of the Hessian
        start = time.time()
        for i in range(self.n_params):
            # for loop over chunks of the dataset

            tangents = t.zeros(self.n_params, device='cuda')
            tangents[i] = 1
            tangents = tuple(self.model.shape_vec_as_params_no_names(tangents))

            total_col = t.zeros(self.n_params, device='cuda')

            for k in range(self.n_chunks):
                j0, j1 = k*self.batch_chunksize, (k+1)*self.batch_chunksize

                # send data to gpu
                data_x = self.data_x[j0:j1].cuda()
                data_y = self.data_y[j0:j1].cuda()

                # defining the function to send to hvp
                def fn_to_optim(*x):

                    z = [{name: p} for (p, name) in zip(x, self.model.param_names)]

                    preds = functional_call(self.model, z, data_x)
                    return self.loss_fn(preds, data_y)

                # calling hvp with the right vector, add to result
                hessian_col = vhp(fn_to_optim, self.params, tangents)[1]
                total_col += (j1-j0) * t.cat([x.flatten() for x in hessian_col])

            # bring it back to cpu
            total_hess.append(total_col.cpu())
            if i%10 == 0 and i != 0:
                stop = time.time()

                time_remaining = (stop - start) * ((self.n_params - i) / i)
                print(f"finished {i}/{self.n_params} --- time remaining:{time_remaining:.1f}s ")

        return t.stack(total_hess, dim=1)


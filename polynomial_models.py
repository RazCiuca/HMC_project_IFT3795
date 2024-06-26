"""
Defines a polynomial regression model of arbitrary degree in pytorch

"""

import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations_with_replacement

class PolynomialRegressor(nn.Module):
    """
    """
    def __init__(self, n_in, n_out, poly_degree, avg_pool_size=None):
        super(PolynomialRegressor, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.poly_degree = poly_degree
        self.poly_indices = []
        self.n_var = 0  # number of variables in the layer before the last

        self.avg_pool = None if avg_pool_size is None else nn.AvgPool2d(kernel_size=avg_pool_size, stride=avg_pool_size)

        # will contain for each degree of the polynomial, a list of tuples corresponding to the indices to gather
        self.indices_to_gather = []

        self.n_predictors = 0

        for degree in range(1, poly_degree+1):

            degree_indices = list(combinations_with_replacement( list(range(self.n_in)), degree) )
            self.indices_to_gather.append(t.LongTensor(degree_indices))
            self.n_predictors += len(self.indices_to_gather[-1])

        self.W = nn.Parameter(t.randn(self.n_predictors, self.n_out)/self.n_predictors**0.5)
        self.b = nn.Parameter(t.zeros(self.n_out))

        self.compute_param_shapes()

    def forward(self, x):
        batch_size = x.size(0)

        # flatten all dimensions but the first one
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        x = x.flatten(1, -1)

        z = []

        for indices in self.indices_to_gather:
            # gather all the indices, multiply them together, and append to z
            z.append(t.prod(x[:, indices], dim=2))

        total_x = t.cat(z, dim=1)

        return total_x @ self.W + self.b


    def compute_param_shapes(self):
        self.param_shapes = [x[1].shape for x in self.named_parameters()]
        self.n_params = sum([np.prod(x) for x in self.param_shapes])
        self.param_names = [x[0] for x in self.named_parameters()]

    def get_vectorized_params(self):
        return t.cat([x[1].flatten() for x in self.named_parameters()])

    def shape_vec_as_params(self, vec):
        params = []
        index = 0

        for x_shape, name in zip(self.param_shapes, self.param_names):
            n = np.prod(x_shape)
            params.append({name: vec[index:n+index].reshape(x_shape)})
            index += n

        return params

    def shape_vec_as_params_no_names(self, vec):
        params = []
        index = 0

        for x_shape, name in zip(self.param_shapes, self.param_names):
            n = np.prod(x_shape)
            params.append(vec[index:n + index].reshape(x_shape))
            index += n

        return params


if __name__ == "__main__":

    # testing stuff
    n_in = 32*32*3
    n_out = 10

    X = t.randn(128, n_in)

    quad_model = PolynomialRegressor(n_in, n_out, poly_degree=2)

    print(quad_model.get_vectorized_params().size())


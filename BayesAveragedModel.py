import numpy as np
import torch as t
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, vmap

class BayesAveragedModel(nn.Module):
    """
    This class takes in a set of parameters of a model, and computes the averaged predictions
    over all those parameters, with a vectorized call
    """
    def __init__(self, model, params, end_softmax=True):
        super(BayesAveragedModel, self).__init__()
        self.model = model
        self.params = params  # dim [n_sample, n_params]
        self.end_softmax = end_softmax

    def forward(self, data_x, chunk_size=16):

        with t.no_grad():

            def to_vectorize(x):
                z = self.model.shape_vec_as_params(x)
                preds = functional_call(self.model, z, data_x)

                return preds

            vectorized_model = vmap(to_vectorize, in_dims=0, chunk_size=chunk_size)

            # dims [n_param_samples, n_batch, dim_out]
            outputs = vectorized_model(self.params)
            if self.end_softmax:
                outputs = t.softmax(outputs, dim=2)

            outputs = outputs.mean(0)

        return outputs

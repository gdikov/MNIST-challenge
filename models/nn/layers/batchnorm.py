import numpy as np


import numpy as np

from layer import AbstractLayer


class Batchnorm(AbstractLayer):
    def __init__(self, incoming, bn_params):
        super(Batchnorm, self).__init__(incoming)
        self.cache = dict()
        self.bn_params = bn_params
        self.init_params()


    def output_shape(self):
        incoming_shape = self.incoming.output_shape()
        return incoming_shape


    def init_params(self):
        self.params = None
        self.dparams = dict()


    def forward(self, x, gamma, beta, bn_param, mode='train'):
        """
        Forward pass for batch normalization.
        """
        eps = self.bn_params.get('eps', 1e-5)
        momentum = self.bn_params.get('momentum', 0.9)

        N, D = x.shape
        running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
        running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

        if mode == 'train':
            sample_mean = np.mean(x, axis=0)
            sample_var = np.mean((x - sample_mean) ** 2, axis=0)
            running_mean = momentum * running_mean + (1 - momentum) * sample_mean
            running_var = momentum * running_var + (1 - momentum) * sample_var
            centered_x = x - sample_mean
            normalization_factor = np.sqrt(sample_var + eps)
            normalized_x = centered_x / normalization_factor
            out = gamma * normalized_x + beta
            self.cache = (x, normalized_x, centered_x, normalization_factor, gamma)
        elif mode == 'test':
            centered_x = x - running_mean
            normalization_factor = np.sqrt(running_var + eps)
            normalized_x = centered_x / normalization_factor

            out = gamma * normalized_x + beta
            self.cache = (x, normalized_x, centered_x, normalization_factor, gamma)
        else:
            raise ValueError('Invalid forward batchnorm shape_mode "%s"' % mode)

        # Store the updated running means back into bn_param
        self.bn_params['running_mean'] = running_mean
        self.bn_params['running_var'] = running_var

        return out


    def backward(self, upstream_derivatives):
        """
        Backward pass for batch normalization.
        """
        N, D = upstream_derivatives.shape
        x, normalized_x, centered_x, normalization_factor, gamma = self.cache['X'], self.cache['norm_x'], \
                                                                   self.cache['center_x'], self.cache['norm_factor'], \
                                                                   self.cache['gamma']

        dx_normalized = gamma * upstream_derivatives
        dvar = np.sum(-0.5 * dx_normalized * (centered_x / (normalization_factor ** 3)), axis=0)
        dmean = np.sum(-dx_normalized / normalization_factor, axis=0) - 2 * dvar * np.mean(centered_x)

        self.dparams['X'] = dx_normalized / normalization_factor + (2.0 / N) * dvar * centered_x + dmean / N
        self.dparams['gamma'] = np.sum(normalized_x.T * upstream_derivatives.T, axis=1)
        self.dparams['beta'] = np.sum(upstream_derivatives, axis=0)

        return self.dparams

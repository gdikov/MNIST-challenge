import numpy as np

from models.nn.layers.layer import AbstractLayer
import models.nn.config as cfg

class Linear(AbstractLayer):


    def __init__(self, incoming, num_units=10):
        super(Linear, self).__init__(incoming)
        self.num_units = num_units
        self.cache = dict()
        self.init_params()


    def output_shape(self):
        return (cfg.batch_size, self.num_units)


    def init_params(self):
        incomig_outputs = np.prod(self.incoming.output_shape()[1:])
        self.params = {'W': 1e-3 * np.random.randn(self.num_units, incomig_outputs),
                       'b': np.zeros(self.num_units)}
        self.dparams = {'W': np.zeros((self.num_units, incomig_outputs)),
                        'b': np.zeros(self.num_units),
                        'X': None}
        self.intit_solvers()


    def forward(self, X, mode='train'):
        """
        Computes the forward pass for an affine (fully-connected) layer.

        The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
        examples, where each example x[i] has shape (d_1, ..., d_k).
        """

        out = X.reshape((X.shape[0], np.prod(X.shape[1:]))).dot(self.params['W'].T) + self.params['b']
        if mode == 'train':
            self.cache['X'] = X

        return out


    def backward(self, upstream_derivatives):
        """
        Computes the backward pass for an affine layer.
        """
        X = self.cache['X']

        self.dparams['X'] = np.dot(upstream_derivatives, self.params['W']).reshape(X.shape)
        self.dparams['W'] = np.dot(upstream_derivatives.T, X.reshape((X.shape[0], np.prod(X.shape[1:]))))
        self.dparams['b'] = np.sum(upstream_derivatives, axis=0)

        return self.dparams['X']

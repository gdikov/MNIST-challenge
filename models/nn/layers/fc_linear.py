import numpy as np

from layer import AbstractLayer
import config as cfg

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


    def forward(self, X):
        """
        Computes the forward pass for an affine (fully-connected) layer.

        The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
        examples, where each example x[i] has shape (d_1, ..., d_k). We will
        reshape each input into a vector of dimension D = d_1 * ... * d_k, and
        then transform it to an output vector of dimension M.

        Inputs:
        - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
        - w: A numpy array of weights, of shape (D, M)
        - b: A numpy array of biases, of shape (M,)

        Returns a tuple of:
        - out: output, of shape (N, M)
        - cache: (x, w, b)
        """

        out = X.reshape((X.shape[0], np.prod(X.shape[1:]))).dot(self.params['W'].T) + self.params['b']
        self.cache['X'] = X

        return out


    def backward(self, upstream_derivatives):
        """
        Computes the backward pass for an affine layer.

        Inputs:
        - dout: Upstream derivative, of shape (N, M)
        - cache: Tuple of:
          - x: Input data, of shape (N, d_1, ... d_k)
          - w: Weights, of shape (D, M)

        Returns a tuple of:
        - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
        - dw: Gradient with respect to w, of shape (D, M)
        - db: Gradient with respect to b, of shape (M,)
        """
        X = self.cache['X']

        self.dparams['X'] = np.dot(upstream_derivatives, self.params['W']).reshape(X.shape)
        self.dparams['W'] = np.dot(upstream_derivatives.T, X.reshape((X.shape[0], np.prod(X.shape[1:]))))
        self.dparams['b'] = np.sum(upstream_derivatives, axis=0)

        return self.dparams['X']

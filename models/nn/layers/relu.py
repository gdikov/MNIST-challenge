import numpy as np

from layer import AbstractLayer


class ReLU(AbstractLayer):
    def __init__(self, incoming, use_leaky=False):
        super(ReLU, self).__init__(incoming)
        self.cache = dict()
        self.use_leaky = use_leaky
        self.init_params()


    def output_shape(self):
        incoming_shape = self.incoming.output_shape()
        return incoming_shape


    def init_params(self):
        self.params = None
        self.dparams = dict()
        self.intit_solvers()


    def forward(self, X, mode='train'):
        """
        Computes the forward pass for a layer of rectified linear units (ReLUs).

        Input:
        - x: Inputs, of any shape

        Returns a tuple of:
        - out: Output, of the same shape as x
        - cache: x
        """
        if not self.use_leaky:
            out = np.maximum(0, X)
        else:
            out = np.maximum(1e-2 * X, X)

        if mode == 'train':
            self.cache['X'] = X
        return out


    def backward(self, upstream_derivatives):
        """
        Computes the backward pass for a layer of rectified linear units (ReLUs).

        Input:
        - dout: Upstream derivatives, of any shape
        - cache: Input x, of same shape as dout

        Returns:
        - dx: Gradient with respect to x
        """
        x = self.cache['X']
        dx = upstream_derivatives
        if not self.use_leaky:
            dx[x <= 0] = 0
        else:
            dx[x <= 0] *= 1e-2
        self.dparams['X'] = dx
        return self.dparams['X']

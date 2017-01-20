import numpy as np

from layer import AbstractLayer


class Dropout(AbstractLayer):
    def __init__(self, incoming, p=None):
        super(Dropout, self).__init__(incoming)
        self.cache = dict()
        self.p = p
        self.init_params()


    def output_shape(self):
        incoming_shape = self.incoming.output_shape()
        return incoming_shape


    def init_params(self):
        self.params = None
        self.dparams = dict()


    def forward(self, X, mode='train'):
        """
        Performs the forward pass for (inverted) dropout.

        Inputs:
        - x: Input data, of any shape
        - dropout_param: A dictionary with the following keys:
          - p: Dropout parameter. We drop each neuron output with probability p.
          - shape_mode: 'test' or 'train'. If the shape_mode is train, then perform dropout;
            if the shape_mode is test, then just return the input.
          - seed: Seed for the random number generator. Passing seed makes this
            function deterministic, which is needed for gradient checking but not in
            real networks.

        Outputs:
        - out: Array of the same shape as x.
        - cache: A tuple (dropout_param, mask). In training shape_mode, mask is the dropout
          mask that was used to multiply the input; in test shape_mode, mask is None.
        """

        mask = None
        out = None

        if mode == 'train':
            # need to multiply by the inverse probability to increase the activation score
            mask = (np.random.rand(*X.shape) < self.p) / self.p
            out = X * mask
        elif mode == 'test':
            out = X

        self.cache = mask
        out = out.astype(X.dtype, copy=False)

        return out


    def backward(self, upstream_derivatives):
        """
        Perform the backward pass for (inverted) dropout.

        Inputs:
        - dout: Upstream derivatives, of any shape
        - cache: (dropout_param, mask) from dropout_forward.
        """
        mask = self.cache
        self.dparams['X'] = upstream_derivatives * mask
        return self.dparams['X']



import numpy as np

from layer import AbstractLayer


class Dropout(AbstractLayer):
    def __init__(self, incoming, dropout_params=None):
        super(Dropout, self).__init__(incoming)
        self.cache = dict()
        self.dropout_params = dropout_params
        self.init_params()


    def output_shape(self):
        incoming_shape = self.incoming.output_shape()
        return incoming_shape


    def init_params(self):
        self.params = None
        self.dparams = dict()


    def forward(self, X):
        """
        Performs the forward pass for (inverted) dropout.

        Inputs:
        - x: Input data, of any shape
        - dropout_param: A dictionary with the following keys:
          - p: Dropout parameter. We drop each neuron output with probability p.
          - mode: 'test' or 'train'. If the mode is train, then perform dropout;
            if the mode is test, then just return the input.
          - seed: Seed for the random number generator. Passing seed makes this
            function deterministic, which is needed for gradient checking but not in
            real networks.

        Outputs:
        - out: Array of the same shape as x.
        - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
          mask that was used to multiply the input; in test mode, mask is None.
        """
        p, mode = self.dropout_params['p'], self.dropout_params['mode']
        if 'seed' in self.dropout_params:
            np.random.seed(self.dropout_params['seed'])

        mask = None
        out = None

        if mode == 'train':
            # need to multiply by the inverse probability to increase the activation score
            mask = np.random.choice([0, 1], size=(X.shape[1],),
                                    p=[1 - p, p]) / p  # np.random.binomial(1, p, size=(x.shape[1],)) / p
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
        mode = self.dropout_params['mode']

        if mode == 'train':
            self.dparams['X'] = upstream_derivatives * mask
        elif mode == 'test':
            self.dparams['X'] = upstream_derivatives
        return self.dparams



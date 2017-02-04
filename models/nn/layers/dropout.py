import numpy as np

from models.nn.layers.layer import AbstractLayer


class Dropout(AbstractLayer):
    def __init__(self, incoming, p=0.5):
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
        Performs the forward pass for dropout.
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
        Perform the backward pass for dropout.
        """
        mask = self.cache
        self.dparams['X'] = upstream_derivatives * mask
        return self.dparams['X']



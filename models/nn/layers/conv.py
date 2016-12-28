import models.nn.layers.conv_utils as helpers
import numpy as np


from layer import AbstractLayer

class Conv(AbstractLayer):
    def __init__(self, incoming, num_filters=32, filter_size=3, conv_params=None):
        super(Conv, self).__init__(incoming)
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.cache = dict()
        self.conv_params = conv_params
        self.init_params()

    def init_params(self):
        num_incomig_channel = self.incoming.output_shape()[0]
        self.params = {'W': 1e-3 * np.random.randn(self.num_filters, num_incomig_channel,
                                                   self.filter_size, self.filter_size),
                       'b': np.zeros(self.num_filters)}
        self.dparams = {'dW': np.zeros(self.num_filters, self.incoming.num_units),
                        'db': np.zeros(self.num_filters),
                        'dX': None}


    def forward(self, X):
        """
        A naive implementation of the forward pass for a convolutional layer.

        The input consists of N data points, each with C channels, height H and width
        W. We convolve each input with F different filters, where each filter spans
        all C channels and has height HH and width WW.

        Input:
        - x: Input data of shape (N, C, H, W)
        - w: Filter weights of shape (F, C, HH, WW)
        - b: Biases, of shape (F,)
        - conv_param: A dictionary with the following keys:
          - 'stride': The number of pixels between adjacent receptive fields in the
            horizontal and vertical directions.
          - 'pad': The number of pixels that will be used to zero-pad the input.

        Returns a tuple of:
        - out: Output data, of shape (N, F, H', W') where H' and W' are given by
          H' = 1 + (H + 2 * pad - HH) / stride
          W' = 1 + (W + 2 * pad - WW) / stride
        - cache: (x, w, b, conv_param)
        """
        N, C, H, W = X.shape
        F, _, HH, WW = self.params['W'].shape
        padding = self.conv_params['pad']
        stride = self.conv_params['stride']
        H_out = 1 + (H + 2 * padding - HH) / stride
        W_out = 1 + (W + 2 * padding - WW) / stride
        out = np.zeros((N, F, H_out, W_out))

        # for each example
        for n in xrange(N):
            # add the padding to the image.
            padded_image = helpers.pad_image(X[n, :, :, :], padding)
            # apply each filter on the input image
            for f in xrange(F):
                helpers.convolve_img(image=padded_image, result=out[n, f, :, :],
                                     kernel=self.params['W'][f, :, :, :], bias=self.params['b'][f],
                                     stride=stride)

        self.cache['X'] = X

        return out


    def backward(self, dout):
        """
        A naive implementation of the backward pass for a convolutional layer.

        Inputs:
        - dout: Upstream derivatives.
        - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

        Returns a tuple of:
        - dx: Gradient with respect to x
        - dw: Gradient with respect to w
        - db: Gradient with respect to b
        """
        x, w, b, padding, stride = self.cache['X'], self.params['W'], self.params['b'], \
                                   self.conv_params['pad'], self.conv_params['stride']
        N = x.shape[0]
        F = w.shape[0]

        dx = np.zeros_like(x)
        for n in xrange(N):
            for f in xrange(F):
                helpers.cumulative_partial_x(weights=w[f, :, :, :], dimage=dx[n, :, :, :], dout=dout[n, f, :, :],
                                             stride=stride, padding=padding)

        dw = np.zeros_like(w)
        for n in xrange(N):
            padded_image = helpers.pad_image(x[n, :, :, :], padding)
            for f in xrange(F):
                helpers.cumulative_partial_w(image=padded_image, dout=dout[n, f, :, :], dw=dw[f, :, :, :], stride=stride)

        db = np.sum(dout, axis=(0, 2, 3))

        self.dparams['dX'] = dx
        self.dparams['dW'] = dw
        self.dparams['db'] = db

        return self.dparams
import conv_utils as helpers
import numpy as np

from layer import AbstractLayer
import models.nn.config as cfg

class Conv(AbstractLayer):
    def __init__(self, incoming, num_filters=32, conv_params=None, conv_mode='scipy'):
        super(Conv, self).__init__(incoming)
        self.num_filters = num_filters
        self.cache = dict()
        self.conv_mode = conv_mode
        if conv_params is None:
            self.conv_params = {'stride': 1, 'pad': (3 - 1) / 2, 'filter_size': 3}
        else:
            self.conv_params = conv_params
        self.init_params()


    def output_shape(self):
        image_height = self.incoming.output_shape()[-2]
        image_width = self.incoming.output_shape()[-1]
        h_out = 1 + (image_height + 2 * self.conv_params['pad'] - self.conv_params['filter_size']) \
                    / self.conv_params['stride']
        w_out = 1 + (image_width + 2 * self.conv_params['pad'] - self.conv_params['filter_size']) \
                    / self.conv_params['stride']
        return (cfg.batch_size, self.num_filters, h_out, w_out)


    def init_params(self):
        num_incoming_channel = self.incoming.output_shape()[1]
        self.params = {'W': 1e-3 * np.random.randn(self.num_filters, num_incoming_channel,
                                                   self.conv_params['filter_size'], self.conv_params['filter_size']),
                       'b': np.zeros(self.num_filters)}
        self.dparams = {'W': np.zeros((self.num_filters, num_incoming_channel,
                                       self.conv_params['filter_size'], self.conv_params['filter_size'])),
                        'b': np.zeros(self.num_filters),
                        'X': None}
        self.intit_solvers()


    def forward(self, X, mode='train'):
        """
        A naive implementation of the forward pass for a convolutional layer.

        The input consists of N data points, each with C channels, height H and width
        W. Each input is convolved with F different filters, where each filter spans
        all C channels and has height HH and width WW.
        """
        N, C, H, W = X.shape
        F, _, HH, WW = self.params['W'].shape
        padding = self.conv_params['pad']
        stride = self.conv_params['stride']
        H_out = 1 + (H + 2 * padding - HH) / stride
        W_out = 1 + (W + 2 * padding - WW) / stride
        out = np.zeros((N, F, H_out, W_out))

        if self.conv_mode == 'naive':
            # for each example
            for n in xrange(N):
                # add the padding to the image.
                padded_image = helpers.pad_image(X[n, :, :, :], padding)
                # apply each filter on the input image
                for f in xrange(F):
                    helpers.convolve_img_naive(image=padded_image, result=out[n, f, :, :],
                                         kernel=self.params['W'][f, :, :, :], bias=self.params['b'][f],
                                         stride=stride)

        elif self.conv_mode == 'scipy':
            for n in xrange(N):
                padded_image = helpers.pad_image(X[n, :, :, :], padding)
                for f in xrange(F):
                    helpers.convolve_img_scipy(image=padded_image, result=out[n, f, :, :],
                                               kernel=self.params['W'][f, :, :, :], bias=self.params['b'][f],
                                               stride=stride)

        if mode == 'train':
            self.cache['X'] = X

        return out


    def backward(self, upstream_derivatives):
        """
        A naive implementation of the backward pass for a convolutional layer.
        """
        x, w, b, padding, stride = self.cache['X'], self.params['W'], self.params['b'], \
                                   self.conv_params['pad'], self.conv_params['stride']
        N = x.shape[0]
        F = w.shape[0]

        dx = np.zeros_like(x)
        for n in xrange(N):
            for f in xrange(F):
                helpers.cumulative_partial_x(weights=w[f, :, :, :],
                                             dimage=dx[n, :, :, :],
                                             dout=upstream_derivatives[n, f, :, :],
                                             stride=stride, padding=padding)

        dw = np.zeros_like(w)
        for n in xrange(N):
            padded_image = helpers.pad_image(x[n, :, :, :], padding)
            for f in xrange(F):
                helpers.cumulative_partial_w(image=padded_image,
                                             dout=upstream_derivatives[n, f, :, :],
                                             dw=dw[f, :, :, :],
                                             stride=stride)

        db = np.sum(upstream_derivatives, axis=(0, 2, 3))

        self.dparams['X'] = dx
        self.dparams['W'] = dw
        self.dparams['b'] = db

        return self.dparams['X']

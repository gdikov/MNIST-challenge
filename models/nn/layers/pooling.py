import numpy as np

from layer import AbstractLayer
import models.nn.config as cfg


class Pool(AbstractLayer):
    def __init__(self, incoming, pool_params=None):
        super(Pool, self).__init__(incoming)
        self.cache = dict()
        self.pool_params = pool_params
        self.init_params()


    def output_shape(self):
        image_height = self.incoming.output_shape()[-2]
        image_width = self.incoming.output_shape()[-1]
        num_incoming_channels = self.incoming.output_shape()[1]
        h_out = (image_height - self.pool_params['pool_height']) / self.pool_params['stride'] + 1
        w_out = (image_width - self.pool_params['pool_width']) / self.pool_params['stride'] + 1
        return (cfg.batch_size, num_incoming_channels, h_out, w_out)


    def init_params(self):
        self.params = None
        self.dparams = None

    def forward(self, X):
        """
        A naive implementation of the forward pass for a max pooling layer.

        Inputs:
        - x: Input data, of shape (N, C, H, W)
        - pool_param: dictionary with the following keys:
          - 'pool_height': The height of each pooling region
          - 'pool_width': The width of each pooling region
          - 'stride': The distance between adjacent pooling regions

        Returns a tuple of:
        - out: Output data
        - cache: (x, pool_param)
        """
        pool_height = self.pool_params['pool_height']
        pool_width = self.pool_params['pool_width']
        stride = self.pool_params['stride']

        N, C, H_image, W_image = X.shape

        H_out = (H_image - pool_height) / stride + 1
        W_out = (W_image - pool_width) / stride + 1

        out = np.zeros((N, C, H_out, W_out))

        for n in xrange(N):
            for h in xrange(0, H_image - pool_height + stride, stride):
                for w in xrange(0, W_image - pool_width + stride, stride):
                    # reshape the patch and perform the pooling on the patch
                    patch = X[n, :, h:h + pool_height, w:w + pool_width].reshape((C, pool_width * pool_height))
                    out[n, :, h / stride, w / stride] = np.max(patch, axis=1)

        self.cache = X
        return out

    def backward(self, upstream_derivatives):
        """
        A naive implementation of the backward pass for a max pooling layer.

        Inputs:
        - dout: Upstream derivatives
        - cache: A tuple of (x, pool_param) as in the forward pass.

        Returns:
        - dx: Gradient with respect to x
        """
        x = self.cache['X']

        pool_height = self.pool_params['pool_height']
        pool_width = self.pool_params['pool_width']
        stride = self.pool_params['stride']

        N, C, H, W = x.shape

        self.dparams['X'] = np.zeros_like(x)

        for n in xrange(N):
            for h in xrange(0, H - pool_height + stride, stride):
                for w in xrange(0, W - pool_width + stride, stride):
                    patch = x[n, :, h:h + pool_height, w:w + pool_width].reshape((C, pool_width * pool_height))
                    mask = (patch.T - np.max(patch, axis=1)).T >= 0
                    derivative_patch = (mask.T * upstream_derivatives[n, :, h / stride, w / stride]).T
                    self.dparams['X'][n, :, h:h + pool_height, w:w + pool_width] = \
                        derivative_patch.reshape((C, pool_width, pool_height))

        return self.dparams['X']

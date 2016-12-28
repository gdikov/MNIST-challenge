import numpy as np

def max_pool_forward_naive(x, pool_param):
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
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    N, C, H_image, W_image = x.shape

    H_out = (H_image - pool_height) / stride + 1
    W_out = (W_image - pool_width) / stride + 1

    out = np.zeros((N, C, H_out, W_out))

    for n in xrange(N):
        for h in xrange(0, H_image - pool_height + stride, stride):
            for w in xrange(0, W_image - pool_width + stride, stride):
                # reshape the patch and perform the pooling on the patch
                patch = x[n, :, h:h + pool_height, w:w + pool_width].reshape((C, pool_width * pool_height))
                out[n, :, h / stride, w / stride] = np.max(patch, axis=1)

    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    x, pool_param = cache

    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    N, C, H, W = x.shape

    dx = np.zeros_like(x)

    for n in xrange(N):
        for h in xrange(0, H - pool_height + stride, stride):
            for w in xrange(0, W - pool_width + stride, stride):
                patch = x[n, :, h:h + pool_height, w:w + pool_width].reshape((C, pool_width * pool_height))
                mask = (patch.T - np.max(patch, axis=1)).T >= 0
                derivative_patch = (mask.T * dout[n, :, h / stride, w / stride]).T
                dx[n, :, h:h + pool_height, w:w + pool_width] = derivative_patch.reshape((C, pool_width, pool_height))
    return dx
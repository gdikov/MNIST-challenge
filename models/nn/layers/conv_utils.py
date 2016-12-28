import numpy as np


def pad_image(image, padding):
    # pad along each chanel but not along the chanel array axis
    return np.pad(image, pad_width=((0, 0), (padding, padding), (padding, padding)), mode='constant',
                  constant_values=(0, 0))


def unpad_image(image, padding):
    # unpad along each chanel but not along the chanel array axis
    return image[:, padding:-padding, padding:-padding]


def convolve_img(image=None, result=None, kernel=None, bias=0, stride=0):
    H_out, W_out = result.shape
    _, H_image, W_image = image.shape
    _, H_kernel, W_kernel = kernel.shape

    for h in xrange(0, H_image - H_kernel + stride, stride):
        for w in xrange(0, W_image - W_kernel + stride, stride):
            # crop a patch from the image (take all the color channels)
            patch = image[:, h:h + H_kernel, w:w + W_kernel]
            # do the dot product, add the bias and store the result
            # here the patch and the kernel are 3-dimensional and the output is 1-dimensional
            result[h / stride, w / stride] = np.sum(patch * kernel) + bias


def cumulative_partial_w(image=None, dout=None, dw=None, stride=0):
    H_out, W_out = dout.shape
    _, H_image, W_image = image.shape
    _, H_kernel, W_kernel = dw.shape

    dw_one_example = np.zeros_like(dw)

    for h in xrange(0, H_image - H_kernel + stride, stride):
        for w in xrange(0, W_image - W_kernel + stride, stride):
            # crop a patch from the image
            patch = image[:, h:h + H_kernel, w:w + W_kernel]
            # multiply by the upstream derivative and accumulate in dw
            dw_one_example += patch * dout[h / stride, w / stride]

    dw += dw_one_example


def cumulative_partial_x(weights=None, dimage=None, dout=None, stride=0, padding=0):
    H_out, W_out = dout.shape
    C, H_image, W_image = dimage.shape
    # compensate for the padding, later will be unpadded
    H_image += 2 * padding
    W_image += 2 * padding
    _, H_kernel, W_kernel = weights.shape

    dimage_one_example = np.zeros((C, H_image, W_image))

    for h in xrange(0, H_image - H_kernel + stride, stride):
        for w in xrange(0, W_image - W_kernel + stride, stride):
            # multiply by the upstream derivative and accumulate in dimage
            dimage_one_example[:, h:h + H_kernel, w:w + W_kernel] += weights * dout[h / stride, w / stride]

    dimage += unpad_image(dimage_one_example, padding)








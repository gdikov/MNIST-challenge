import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def plot_digits(data, labels, plot_shape):
    # Get the figure and axes.
    num_subplots = np.prod(plot_shape)
    fig, axes = plt.subplots(*plot_shape)
    axes = axes.reshape(num_subplots)
    fig.suptitle("MNIST digits with labels")

    for axis, index in zip(axes, np.arange(data.shape[0])[:num_subplots]):
        image = data[index, :, :]
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)
        axis.imshow(image, cmap=plt.cm.Greys_r)
        axis.title.set_text('{}'.format(int(labels[index])))
    plt.show()

def plot_filters(filters, plot_shape, channel=1):
    # Get the figure and axes.
    num_subplots = np.prod(plot_shape)
    fig, axes = plt.subplots(*plot_shape)
    axes = axes.reshape(num_subplots)
    fig.suptitle("Conv filters")

    for axis, index in zip(axes, np.arange(filters.shape[0])[:num_subplots]):
        image = filters[index, channel-1, :, :]
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)
        axis.imshow(image, cmap=plt.cm.Greys_r, interpolation='none')
    plt.show()
"""
GENERAL NOTE:
    These methods are adapted from the source accompanying the assignments in the online free course cs231n by A. Karpathy.
    Link to the course: http://cs231n.stanford.edu
    Although the content of the methods is largely changed, I decided to cite the source as I adapted the author's
    logic and structure.

    load_MNIST_from_raw(..) is taken from http://g.sweyla.com/blog/2012/mnist-numpy/
    from the reference: https://raw.githubusercontent.com/amitgroup/amitgroup/master/amitgroup/io/mnist.py

DESCRIPTION:
    Main functionality of this module is to load and store the MNIST dataset.
"""

import cPickle as pickle
import numpy as np
import os
import struct
from array import array as pyarray


def load_MNIST(num_training=50000, num_validation=10000, num_test=10000, force_split=False):
    """
    Load the MNIST dataset from disk and perform preprocessing to prepare
    it for classifiers. These are the same steps as we used for the SVM, but
    condensed to a single function.
    """

    # Load the raw MNIST data
    MNIST_path = os.path.join(os.path.realpath(os.path.dirname(__file__)), "../data/")
    path_to_pickles = os.path.join(MNIST_path, 'pickles')
    path_to_train_mask = os.path.join(path_to_pickles, 'train_mask.npy')
    path_to_val_mask = os.path.join(path_to_pickles, 'val_mask.npy')

    X_train, y_train = load_MNIST_from_raw(path=MNIST_path, dataset='training')
    X_test, y_test = load_MNIST_from_raw(path=MNIST_path, dataset='testing')

    val_mask = np.array([])
    if not force_split and os.path.exists(path_to_pickles):
        if num_validation > 0:
            with open(path_to_val_mask, 'rb') as f:
                val_mask = np.load(f)
        with open(path_to_train_mask, 'rb') as f:
            train_mask = np.load(f)
    else:
        if not os.path.exists(path_to_pickles):
            os.makedirs(path_to_pickles)

        num_train_samples = X_train.shape[0]
        # Split and subsample the data
        if num_validation > 0:
            val_mask = np.random.choice(num_train_samples, replace=False, size=num_validation)
        train_mask = np.setdiff1d(np.arange(num_train_samples), val_mask, assume_unique=True)
        with open(path_to_val_mask, 'wb') as f:
            np.save(f, val_mask)
        with open(path_to_train_mask, 'wb') as f:
            np.save(f, train_mask)

    X_val = X_train[val_mask][:num_validation]
    y_val = y_train[val_mask][:num_validation]
    X_train = X_train[train_mask][:num_training]
    y_train = y_train[train_mask][:num_training]
    X_test = X_test[:num_test]
    y_test = y_test[:num_test]

    if num_validation > 0:
        assert X_val.shape[0] + X_train.shape[0] == num_training + num_validation, \
            "Something went worng while splitting the dataset into train and validation subsets"

    # Normalize the data: subtract the mean image
    if num_training > 1:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        if num_validation > 0:
            X_val -= mean_image
        X_test -= mean_image

    print("MNIST dataset is loaded from disk and normalized to {0} mean".format(np.mean(X_train)))

    # Package data into a dictionary
    if num_validation > 0:
        data_dict = {'x_train': X_train, 'y_train': y_train,
                     'x_val': X_val, 'y_val': y_val,
                     'x_test': X_test, 'y_test': y_test}
    else:
        data_dict = {'x_train': X_train, 'y_train': y_train,
                     'x_test': X_test, 'y_test': y_test}
    return data_dict


def load_MNIST_from_raw(dataset="training", digits=None,
                        path=None, asbytes=False,
                        selection=None, return_labels=True,
                        return_indices=False):
    """
    Loads MNIST files into a 3D numpy array.

    You have to download the data separately from [MNIST]_. It is recommended
    to set the environment variable ``MNIST`` to point to the folder where you
    put the data, so that you don't have to select path. On a Linux+bash setup,
    this is done by adding the following to your ``.bashrc``::

        export MNIST=/path/to/mnist

    Parameters
    ----------
    dataset : str
        Either "training" or "testing", depending on which dataset you want to
        load.
    digits : list
        Integer list of digits to load. The entire database is loaded if set to
        ``None``. Default is ``None``.
    path : str
        Path to your MNIST datafiles. The default is ``None``, which will try
        to take the path from your environment variable ``MNIST``. The data can
        be downloaded from http://yann.lecun.com/exdb/mnist/.
    asbytes : bool
        If True, returns data as ``numpy.uint8`` in [0, 255] as opposed to
        ``numpy.float64`` in [0.0, 1.0].
    selection : slice
        Using a `slice` object, specify what subset of the dataset to load. An
        example is ``slice(0, 20, 2)``, which would load every other digit
        until--but not including--the twentieth.
    return_labels : bool
        Specify whether or not labels should be returned. This is also a speed
        performance if digits are not specified, since then the labels file
        does not need to be read at all.
    return_indicies : bool
        Specify whether or not to return the MNIST indices that were fetched.
        This is valuable only if digits is specified, because in that case it
        can be valuable to know how far
        in the database it reached.

    Returns
    -------
    images : ndarray
        Image data of shape ``(N, rows, cols)``, where ``N`` is the number of images. If neither labels nor inices are returned, then this is returned directly, and not inside a 1-sized tuple.
    labels : ndarray
        Array of size ``N`` describing the labels. Returned only if ``return_labels`` is `True`, which is default.
    indices : ndarray
        The indices in the database that were returned.

    Examples
    --------
    Assuming that you have downloaded the MNIST database and set the
    environment variable ``$MNIST`` point to the folder, this will load all
    images and labels from the training set:

    >>> images, labels = ag.io.load_mnist('training') # doctest: +SKIP

    Load 100 sevens from the testing set:

    >>> sevens = ag.io.load_mnist('testing', digits=[7], selection=slice(0, 100), return_labels=False) # doctest: +SKIP

    """

    # The files are assumed to have these names and should be found in 'path'
    files = {
        'training': ('train-images-idx3-ubyte', 'train-labels-idx1-ubyte'),
        'testing': ('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte'),
    }

    if path is None:
        try:
            path = os.environ['MNIST']
        except KeyError:
            raise ValueError("Unspecified path requires environment variable $MNIST to be set")

    try:
        images_fname = os.path.join(path, files[dataset][0])
        labels_fname = os.path.join(path, files[dataset][1])
    except KeyError:
        raise ValueError("Data set must be 'testing' or 'training'")

    # We can skip the labels file only if digits aren't specified and labels aren't asked for
    if return_labels or digits is not None:
        flbl = open(labels_fname, 'rb')
        magic_nr, size = struct.unpack(">II", flbl.read(8))
        labels_raw = pyarray("b", flbl.read())
        flbl.close()

    fimg = open(images_fname, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    images_raw = pyarray("B", fimg.read())
    fimg.close()

    if digits:
        indices = [k for k in range(size) if labels_raw[k] in digits]
    else:
        indices = range(size)

    if selection:
        indices = indices[selection]
    N = len(indices)

    images = np.zeros((N, rows, cols), dtype=np.uint8)

    if return_labels:
        labels = np.zeros((N), dtype=np.int8)
    for i, index in enumerate(indices):
        images[i] = np.array(images_raw[ indices[i]*rows*cols : (indices[i]+1)*rows*cols ]).reshape((rows, cols))
        if return_labels:
            labels[i] = labels_raw[indices[i]]

    if not asbytes:
        images = images.astype(float)/255.0

    ret = (images,)
    if return_labels:
        ret += (labels,)
    if return_indices:
        ret += (indices,)
    if len(ret) == 1:
        return ret[0] # Don't return a tuple of one
    else:
        return ret

if __name__ == "__main__":

    dataset = load_MNIST()
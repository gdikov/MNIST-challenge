import numpy as np

class OneHotLabels(object):
    def __init__(self, dim):
        self.dim = dim

    def generate_labels(self, ys):
        num_labels = ys.shape[0]
        one_hot_ys = np.zeros((num_labels, self.dim))
        one_hot_ys[np.arange(num_labels), ys] = 1
        return one_hot_ys


class BinaryLabels(object):

    def __init__(self, class_one):
        self.cls1 = class_one

    def generate_labels(self, ys):
        num_labels = ys.shape[0]
        binary_labels = np.zeros(num_labels)
        binary_labels[ys == self.cls1] = 1
        return binary_labels
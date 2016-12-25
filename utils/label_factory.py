import numpy as np

class OneHot(object):
    def __init__(self, dim):
        self.dim = dim

    def generate_labels(self, ys):
        num_labels = ys.shape[0]
        one_hot_ys = np.zeros((num_labels, self.dim))
        one_hot_ys[np.arange(num_labels), ys] = 1
        return one_hot_ys

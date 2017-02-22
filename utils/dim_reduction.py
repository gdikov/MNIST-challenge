import numpy as np
import scipy.linalg as spl

class DimReduction(object):
    def __init__(self, dim):
        self.new_dimensionality = dim

    def reduce(self, data, **kwargs):
        pass



class PCA(DimReduction):
    def __init__(self, num_components):
        super(PCA, self).__init__(dim=num_components)

    def reduce(self, data, **kwargs):
        """

        :param data: N by D1 x ... x Dk array
        :param kwargs:
        :return: reduced in dimensions array of shape N x D_hat where D_hat <= Prod_i(D_i)
        """
        data = data.reshape(data.shape[0], -1)
        assert data.shape[1] >= self.new_dimensionality, \
            "ERROR: new dimensionality must be lower than the current one"
        mean = np.mean(data, axis=0)
        data_hat = data - mean

        covariance = data_hat.T.dot(data_hat)
        eigvals, eigvecs = spl.eigh(covariance, eigvals=(covariance.shape[0] - self.new_dimensionality,
                                                         covariance.shape[0] - 1))
        data_hat = data_hat.dot(eigvecs)

        show_sample_reconstruction = kwargs.get('show_reconstruction', False)
        if show_sample_reconstruction:
            from utils.vizualiser import plot_digits
            data_hat = data_hat[:64, :].dot(eigvecs.T).reshape(-1, 28, 28)
            plot_digits(data_hat, plot_shape=(8, 8))

        return_reconstructed = kwargs.get('return_reconstructed', False)
        if return_reconstructed:
            data_hat = data_hat.dot(eigvecs.T).reshape(-1, 28, 28)

        return data_hat


if __name__ == "__main__":
    from utils.data_utils import load_MNIST

    data_train, _ = load_MNIST(num_training=64, num_validation=10)

    dr = PCA(num_components=5)
    data_train_hat = dr.reduce(data_train['x_train'])
    from utils.vizualiser import plot_digits
    data_train_hat = data_train_hat.reshape(-1, 8, 8)
    plot_digits(data_train_hat, data_train['y_train'], (8, 8))



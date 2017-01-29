import numpy as np
import scipy as sp
import scipy.sparse as spm
import scipy.linalg as spl
import itertools as it

from models.model import AbstractModel

from distribution_estimation.approximator import LaplaceApproximation

n_classes = 10

class GaussianProcesses(AbstractModel):
    def __init__(self, kernel='sqr_exp', distribution_estimation='analytic'):
        super(GaussianProcesses, self).__init__('GaussianProcesses')
        self.hyper_params = None
        self.kernel = self._build_kernel(kernel)
        self.cov_function = None
        self.mean_function = None
        self.latent_function = None
        if distribution_estimation == 'analytic':
            self.estimator = LaplaceApproximation()
        else:
            # TODO: try with a sampling procedure
            raise NotImplementedError


    def _build_kernel(self, kernel='sqr_exp'):
        def squared_exponential(x_i, x_j):
            d = x_i - x_j
            res = (self.hyper_params['sigma'] ** 2) * \
                   np.exp(-0.5 * np.dot(d.T, d) / self.hyper_params['lambda'])
            return res

        def linear(x_i, x_j):
            return self.hyper_params['sigma'] ** 2 + np.dot(x_i.T, x_j)

        def periodic(x_i, x_j):
            d = np.sin(0.5 * (x_i - x_j))
            return np.exp(-2.0 * np.dot(d.T, d) / (self.hyper_params['lambda']) ** 2)

        if kernel == 'sqr_exp':
            self.hyper_params = {'sigma': 1.0,
                                 'lambda': 2.0}
            return squared_exponential
        elif kernel == 'lin':
            self.hyper_params = {'sigma': 1.0}
            return linear
        elif kernel == 'periodic':
            self.hyper_params = {'lambda': 1.0}
            return periodic
        else:
            raise NotImplementedError


    def _build_cov_function(self, data=None, _class=0):
        if _class == -1:
            print("\t\tComputing the extended covariance function")
            num_train_samples = self.data['x_train'].shape[0]
            num_samples_new = data.shape[0]
            cov_function = np.zeros((num_samples_new, num_train_samples))
            for i in xrange(num_samples_new):
                for j in xrange(num_train_samples):
                    cov_function[i, j] = self.kernel(data[i], self.data['x_train'][j])
            return cov_function
        else:
            print("\t\tComputing covariance function for class {}".format(_class))
            num_samples = data.shape[0]
            cov_function = np.zeros((num_samples, num_samples))
            upper_tri_ids = it.combinations(np.arange(num_samples), 2)
            # compute upper triangular submatrix and then the lower using the symmetry property
            for i, j in upper_tri_ids:
                cov_function[i, j] = self.kernel(data[i], data[j])
            cov_function += cov_function.T
            # compute the diagonal
            for i in xrange(num_samples):
                cov_function[i, i] = self.kernel(data[i], data[i])
            return cov_function


    def _init_gp_functions(self):
        num_samples, dim_x, dim_y = self.data['x_train'].shape
        self.data['x_train'] = self.data['x_train'].reshape(num_samples, dim_x * dim_y)
        # TODO: add or sample hyperparams for the kernel
        self.cov_function = [self._build_cov_function(self.data['x_train'], _class=c) for c in xrange(n_classes)]
        self.latent_function = np.zeros((n_classes, num_samples))
        self.mean_function = np.zeros((n_classes, num_samples))


    def fit(self, train_data, **kwargs):
        """
        Apply multi-class Laplace approximation with a squared expoenntial kernel covariance function.
        Use a shared signal amplitude and length-scale factors sigma and l for all 10 classes (i.e. latent functions)
        and all 28*28 input dimensions.
        :param train_data:
        :param kwargs:
        :return:
        """
        self.data = train_data
        self._init_gp_functions()

        self.f_posterior, approx_log_marg_likelihood = self.estimator.approximate(self.cov_function,
                                                                                  self.data['y_train'])
        # TODO: persist the posterior latent function
        print(approx_log_marg_likelihood)


    def predict(self, new_data):
        num_samples, dim_x, dim_y = new_data.shape
        new_data = new_data.reshape(num_samples, dim_x * dim_y)

        # extend the covariance function
        cov_function_new_hetero = [self._build_cov_function(new_data, _class=-1)
                                   for _ in xrange(n_classes)]
        cov_function_new_auto = [np.array([self.kernel(new_data[i], new_data[i])
                                                    for i in xrange(num_samples)])
                                 for _ in xrange(n_classes)]
        cov_function_test = {'auto': cov_function_new_auto,
                             'hetero': cov_function_new_hetero}
        self.estimator.compute_latent_mean_cov(cov_matrix_train=self.cov_function,
                                               cov_matrix_test=cov_function_test,
                                               f_posterior=self.f_posterior)

        return None


if __name__ == "__main__":
    from utils.data_utils import load_MNIST

    data_train, data_test = load_MNIST(num_training=12, num_validation=5)

    model = GaussianProcesses()

    model.fit(data_train)

    predictions = model.predict(data_train['x_val'])
    #
    # test_acc = np.sum(predictions == data_train['y_val']) / float(predictions.shape[0]) * 100.
    # print("Validation accuracy: {0}"
    #       .format(test_acc))
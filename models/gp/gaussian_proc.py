import numpy as np
import itertools as it

from models.model import AbstractModel

from distribution_estimation.approximator import LaplaceApproximation
from distribution_estimation.sampler import MonteCarlo
from distribution_estimation.kernel_optimiser import HyperParameterOptimiser

n_classes = 10

class GaussianProcess(AbstractModel):
    def __init__(self, kernel='sqr_exp', distribution_estimation='analytic', num_classes=2):
        super(GaussianProcess, self).__init__('GaussianProcesses')
        self.n_classes = num_classes
        self.hyper_params = None
        self.kernel = self.build_kernel(kernel)
        self.cov_function = None
        self.mean_function = None
        self.latent_function = None
        if distribution_estimation == 'analytic':
            self.estimator = LaplaceApproximation()
        else:
            # TODO: try with a sampling procedure
            raise NotImplementedError
        self.sampler = MonteCarlo(num_sampling_steps=100)
        self.optimiser = HyperParameterOptimiser()
        self.num_restarts_fitting = 1



    def build_kernel(self, kernel='sqr_exp'):
        def squared_exponential(x_i, x_j, cls=0):
            d = x_i - x_j
            res = (self.hyper_params['sigma'][cls] ** 2) * \
                  np.exp(-0.5 * np.dot(d.T, d) / self.hyper_params['lambda'][cls])
            return res

        def linear(x_i, x_j, cls=0):
            return self.hyper_params['sigma'][cls] ** 2 + np.dot(x_i.T, x_j)

        def periodic(x_i, x_j, cls=0):
            d = np.sin(0.5 * (x_i - x_j))
            return np.exp(-2.0 * np.dot(d.T, d) / (self.hyper_params['lambda'][cls]) ** 2)

        if kernel == 'sqr_exp': # best for multi: 1 - 1,1 / 16,3, 16.4
            self.hyper_params = {'sigma': np.random.uniform(2.0, 2.1, size=n_classes if self.n_classes > 2 else 1),
                                 'lambda': np.random.uniform(10.3, 10.4, size=n_classes if self.n_classes > 2 else 1)}
            return squared_exponential
        elif kernel == 'lin':
            self.hyper_params = {'sigma': np.random.uniform(0.7, 1.3, size=n_classes if self.n_classes > 2 else 1)}
            return linear
        elif kernel == 'periodic':
            self.hyper_params = {'lambda': np.random.uniform(0.7, 1.3, size=n_classes if self.n_classes > 2 else 1)}
            return periodic
        else:
            raise NotImplementedError

    def build_cov_function(self, data=None, mode='symm'):
        if mode == 'hetero':
            print("\t\tComputing the covariance between training and test samples")
            num_train_samples = self.data['x_train'].shape[0]
            num_samples_new = data.shape[0]
            if self.n_classes > 2:
                cov_function = [np.zeros((num_samples_new, num_train_samples)) for _ in xrange(self.n_classes)]
                for c in xrange(self.n_classes):
                    for i in xrange(num_samples_new):
                        for j in xrange(num_train_samples):
                            cov_function[c][i, j] = self.kernel(data[i], self.data['x_train'][j], cls=c)
            else:
                cov_function = np.zeros((num_samples_new, num_train_samples))
                for i in xrange(num_samples_new):
                    for j in xrange(num_train_samples):
                        cov_function[i, j] = self.kernel(data[i], self.data['x_train'][j])

        elif mode == 'auto':
            print("\t\tComputing the covariance between test samples")
            if self.n_classes > 2:
                cov_function = [np.array([self.kernel(data[i], data[i], cls=c) for i in xrange(data.shape[0])])
                                for c in xrange(n_classes)]
            else:
                cov_function = np.array([self.kernel(data[i], data[i]) for i in xrange(data.shape[0])])

        elif mode == 'symm':
            print("\t\tComputing the covariance between training samples")
            num_samples = data.shape[0]
            upper_tri_ids = it.combinations(np.arange(num_samples), 2)
            if self.n_classes > 2:
                cov_function = [np.zeros((num_samples, num_samples)) for _ in xrange(n_classes)]
                # compute upper triangular submatrix and then the lower using the symmetry property
                for c in xrange(n_classes):
                    for i, j in upper_tri_ids:
                        cov_function[c][i, j] = self.kernel(data[i], data[j], cls=c)
                    cov_function[c] += cov_function[c].T
                    # compute the diagonal
                    for i in xrange(num_samples):
                        cov_function[c][i, i] = self.kernel(data[i], data[i], cls=c)
            else:
                cov_function = np.zeros((num_samples, num_samples))
                for i, j in upper_tri_ids:
                    cov_function[i, j] = self.kernel(data[i], data[j])
                cov_function += cov_function.T
                for i in xrange(num_samples):
                    cov_function[i, i] = self.kernel(data[i], data[i])
        else:
            raise ValueError

        return cov_function

    def set_gp_functions(self, latent_init=None, cov_init=None, mean_init=None):
        if cov_init is None:
            self.cov_function = self.build_cov_function(self.data['x_train'], mode='symm')
        else:
            self.cov_function = cov_init
        if latent_init is None:
            if self.n_classes > 2:
                self.latent_function = np.zeros((n_classes, self.num_samples))
            else:
                self.latent_function = np.zeros(self.num_samples)
        else:
            self.latent_function = latent_init
        if mean_init is None:
            if self.n_classes > 2:
                self.mean_function = np.zeros((n_classes, self.num_samples))
            else:
                self.mean_function = np.zeros(self.num_samples)
        else:
            self.mean_function = mean_init


    def update_hypers_and_functions(self, new_hypers):
        self.hyper_params = new_hypers
        self.cov_function = self.build_cov_function(self.data['x_train'], mode='symm')


    def fit(self, train_data, **kwargs):
        pass

    def predict(self, new_data, **kwargs):
        pass



class MulticlassGaussianProcess(GaussianProcess):
    def __init__(self, kernel='sqr_exp', distribution_estimation='analytic', classification_mode='mixed_binary'):
        super(MulticlassGaussianProcess, self).__init__(kernel=kernel,
                                                        distribution_estimation=distribution_estimation,
                                                        num_classes=n_classes)
        self.classification_mode = classification_mode   # can be 'multi' but it is not finished


    def _stratify_training_data(self, c, size=100):
        ids_of_class_c = np.arange(self.data['y_train'].shape[0])[self.data['y_train'] == c]
        ids_the_rest = np.arange(self.data['y_train'].shape[0])[self.data['y_train'] != c]
        # pick a random subset 50% of size from them
        size_of_class_c = size // 2
        subset_ids_of_class_c = np.random.choice(ids_of_class_c, size=size_of_class_c, replace=False)
        subset_ids_rest = np.random.choice(ids_the_rest, size=size - size_of_class_c, replace=False)
        shuffled_order = np.random.permutation(subset_ids_of_class_c.shape[0] + subset_ids_rest.shape[0])
        data_dict = {'x_train': np.vstack((self.data['x_train'][subset_ids_of_class_c],
                                           self.data['x_train'][subset_ids_rest]))[shuffled_order],
                     'y_train': np.hstack((self.data['y_train'][subset_ids_of_class_c],
                                           self.data['y_train'][subset_ids_rest]))[shuffled_order]}
        return data_dict


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

        self.num_samples, dim_x, dim_y = self.data['x_train'].shape
        self.data['x_train'] = self.data['x_train'].reshape(self.num_samples, dim_x * dim_y)

        if self.classification_mode == 'multi':
            self.set_gp_functions()
            f_posterior, approx_log_marg_likelihood = \
                self.estimator.approximate_multiclass(self.cov_function,
                                                      self.data['y_train'],
                                                      latent_init=self.latent_function)
            # TODO: persist the posterior latent function if an improvement is observed
            # better_hypers = self.optimiser.optimise_hyper_params_multiclass(f_posterior)
            # cov_posterior, mean_posterior = self._recompute_mean_cov(better_hypers)
            # self._set_gp_functions(latent_init=f_posterior, cov_init=cov_posterior, mean_init=mean_posterior)
            self.latent_function = f_posterior
        elif self.classification_mode == 'mixed_binary':
            self.binary_gps = list()
            for cls in range(1, 2):
                train_data_for_c = self._stratify_training_data(c=cls, size=200)
                gp = BinaryGaussianProcessClassifier()
                gp.fit(train_data_for_c)
                self.binary_gps.append((cls, gp))
        else:
            raise ValueError


    def predict(self, new_data, **kwargs):
        if len(new_data.shape) == 3:
            num_samples, dim_x, dim_y = new_data.shape
            new_data = new_data.reshape(num_samples, dim_x * dim_y)
        else:
            num_samples = new_data.shape[0]

        if self.classification_mode == 'multi':
            cov_function_test = {'auto': self.build_cov_function(new_data, mode='auto'),
                                 'hetero': self.build_cov_function(new_data, mode='hetero')}
            latent_mean, latent_cov = self.estimator.compute_latent_mean_cov_multiclass(cov_matrix_train=self.cov_function,
                                                                                        cov_matrix_test=cov_function_test,
                                                                                        f_posterior=self.latent_function)
            predicted_class_probs = self.sampler.sample(latent_mean=latent_mean, latent_cov=latent_cov)
            return np.argmax(predicted_class_probs, axis=1)

        elif self.classification_mode == 'mixed_binary':
            predictions = np.zeros((num_samples, n_classes))
            for cls, gp in self.binary_gps:
                predictions[:, cls] = gp.predict(new_data, return_probs=True)
            prediction_classes = np.argmax(np.vstack((predictions[:, cls], 1-predictions[:, cls])).T, axis=1)
        else:
            raise ValueError

        return prediction_classes



class BinaryGaussianProcessClassifier(GaussianProcess):
    def __init__(self, kernel='sqr_exp', distribution_estimation='analytic', class_positive=0):
        super(BinaryGaussianProcessClassifier, self).__init__(kernel=kernel,
                                                              distribution_estimation=distribution_estimation,
                                                              num_classes=2)
        self.class_positive = class_positive


    def fit(self, train_data, **kwargs):
        self.data = train_data
        if len(self.data['x_train'].shape) == 3:
            self.num_samples, dim_x, dim_y = train_data.shape
            self.data = train_data.reshape(self.num_samples, dim_x * dim_y)
        self.num_samples = self.data['x_train'].shape[0]

        self.set_gp_functions()
        for i in range(1):
            f_posterior, a, approx_log_marg_likelihood = \
                self.estimator.approximate_binary(self.cov_function, self.data['y_train'],
                                                  latent_init=self.latent_function, cls=self.class_positive)
            # better_hypers = self.optimiser.optimise_hyper_params_binary(self.cov_function,
            #                                                             f_posterior,
            #                                                             a, self.data['y_train'],
            #                                                             self.hyper_params, self.class_positive)
            # self.hyper_params = better_hypers
            # self.set_gp_functions(latent_init=f_posterior)  # it also recomputes the new covariance
        self.latent_function = f_posterior


    def predict(self, new_data, **kwargs):
        # extend the covariance function
        cov_function_test = {'auto': self.build_cov_function(new_data, mode='auto'),
                             'hetero': self.build_cov_function(new_data, mode='hetero')}
        latent_mean, latent_cov = self.estimator.compute_latent_mean_cov_binary(cov_matrix_train=self.cov_function,
                                                                                cov_matrix_test=cov_function_test,
                                                                                f_posterior=self.latent_function)
        predict_probs = kwargs.get('return_probs', False)
        predicted_class_probs = self.estimator.approximate_posterior_integral(latent_mean=latent_mean,
                                                                              latent_cov=latent_cov)
        if predict_probs:
            return predicted_class_probs
        else:
            return (predicted_class_probs > 0.5).astype(np.int32)    # return +1 for class_one and 0 otherwise





if __name__ == "__main__":
    from utils.data_utils import load_MNIST

    data_train, data_test = load_MNIST(num_training=10000, num_validation=100)

    model = MulticlassGaussianProcess(classification_mode='mixed_binary')

    model.fit(data_train)

    predictions = model.predict(data_train['x_val'])
    print(predictions)
    print(data_train['y_val'])
    test_acc = np.sum(predictions == data_train['y_val']) / float(predictions.shape[0]) * 100.
    print("Validation accuracy: {0}"
          .format(test_acc))

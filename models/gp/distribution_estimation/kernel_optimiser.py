import scipy.sparse.linalg as spsl
import scipy.linalg as spl
import scipy.sparse as spm
import scipy.optimize as spo
import numpy as np

from numerics.softmax import logistic
from utils.label_factory import BinaryLabels

n_classes = 10

class HyperParameterOptimiser(object):
    def __init__(self):
        pass

    def optimise_hyper_params_multiclass(self, f_posterior):
        raise NotImplementedError
        # # perform standard optimisation routine using the gradients of the likelihood and the likelihood
        # grads_hypers, likelihood_funciton = self._compute_hypers_partial_derivatives_multiclass()


    def _compute_hypers_partial_derivatives_multiclass(self, cov_matrix, f_posterior, targets):
        raise NotImplementedError
        # num_test_samples = targets.shape[0]
        # # W = - grad^2 log p(y|f)
        # w = np.zeros((num_test_samples, n_classes)) # TODO fill with sense
        # w = np.sqrt(w)
        # # L = cholesky(I + W^(1/2) * K * W^(1/2))
        # L = spl.cholesky(spm.identity(num_test_samples) + spm.diags(w).dot())


    def optimise_hyper_params_binary(self, cov_matrix, f_posterior, a, targets, hypers, cls):
        likelihood, delta_sigma, delta_lambda = self._compute_hypers_partial_derivatives_binary(cov_matrix, f_posterior,
                                                                                                a, targets, hypers, cls)
        new_hypers = {'sigma': np.max((hypers['sigma'] + 0.01*delta_sigma, np.array([0.5])), axis=0),
                      'lambda': np.max((hypers['lambda'] + 0.01*delta_lambda, np.array([0.5])), axis=0)}
        print("class {} new: {}".format(cls, new_hypers))
        return new_hypers


    def _compute_hypers_partial_derivatives_binary(self, cov_matrix, f_posterior, a, targets, hypers, cls):
        num_samples = f_posterior.shape[0]
        binary_targets = BinaryLabels(class_one=cls).generate_labels(targets)
        pi = logistic(f_posterior)
        # W = - delta^2 log p(y|f)
        neg_hessian = pi - pi ** 2
        neg_hessian_sqrt = spm.diags(np.sqrt(neg_hessian), format='csc')
        # L = cholesky(I + W^(1/2) * K * W^(1/2))
        L = spl.cholesky((spm.identity(num_samples) +
                          neg_hessian_sqrt.dot(spm.csc_matrix(cov_matrix).dot(neg_hessian_sqrt)))
                         .toarray(), lower=True)
        approx_log_marg_likelihood = -0.5 * a.dot(f_posterior) \
                                     + np.sum(np.log(logistic(f_posterior))) \
                                     - np.sum(np.log(np.diagonal(L)))
        # R = W^(1/2) * L^T \ (L \ W^(1/2))
        R = neg_hessian_sqrt.dot(spsl.spsolve(spm.csc_matrix(L.T), spsl.spsolve(spm.csc_matrix(L), neg_hessian_sqrt)))
        C = spsl.spsolve(spm.csc_matrix(L), neg_hessian_sqrt.dot(cov_matrix))
        delta3_pi = (pi**2) * (1 - pi) - pi * ((1 - pi)**2)
        # -1/2 * diag(diag(K) - diag(C^T * C)) * delta^# log p(y|f)
        s_2 = -0.5 * spm.diags(np.diagonal(cov_matrix) - np.diagonal(C.T.dot(C))).dot(delta3_pi)

        # compute derivative for hyper['sigma']
        derivative_matrix_sigma = 2.0 * cov_matrix / hypers['sigma']
        s_1 = 0.5 * a.dot(derivative_matrix_sigma.dot(a)) - 0.5 * np.trace(R.dot(derivative_matrix_sigma))
        b = derivative_matrix_sigma.dot(binary_targets - pi)
        s_3 = b - cov_matrix.dot(R.dot(b))
        delta_sigma = s_1 + s_2.dot(s_3)

        # compute derivative for hyper['lambda']
        derivative_matrix_lambda = cov_matrix * (-np.log(cov_matrix / (hypers['sigma']**2)) / hypers['lambda'])
        s_1 = 0.5 * a.dot(derivative_matrix_lambda.dot(a)) - 0.5 * np.trace(R.dot(derivative_matrix_sigma))
        b = derivative_matrix_lambda.dot(binary_targets - pi)
        s_3 = b - cov_matrix.dot(R.dot(b))
        delta_lambda = s_1 + s_2.dot(s_3)
        print("apprx: {}".format(approx_log_marg_likelihood))
        return approx_log_marg_likelihood, delta_sigma, delta_lambda



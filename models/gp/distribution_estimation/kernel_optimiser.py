import scipy.sparse.linalg as spsl
import scipy.linalg as spl
import scipy.sparse as spm
import numpy as np

n_classes = 10

class HyperParameterOptimiser(object):
    def __init__(self):
        pass

    def optimise_hyper_params_multiclass(self, f_posterior):
        raise NotImplementedError
        # perform standard optimisation routine using the gradients of the likelihood and the likelihood
        grads_hypers, likelihood_funciton = self._compute_hypers_partial_derivatives_multiclass()


    def _compute_hypers_partial_derivatives_multiclass(self, cov_matrix, f_posterior, targets):
        raise NotImplementedError
        num_test_samples = targets.shape[0]
        # W = - grad^2 log p(y|f)
        w = np.zeros((num_test_samples, n_classes)) # TODO fill with sense
        w = np.sqrt(w)
        # L = cholesky(I + W^(1/2) * K * W^(1/2))
        L = spl.cholesky(spm.identity(num_test_samples) + spm.diags(w).dot())


    def optimise_hyper_params_binary(self, f_posterior):
        grads_hypers, likelihood_funciton = self._compute_hypers_partial_derivatives_multiclass()


    def _compute_hypers_partial_derivatives_bianry(self, cov_matrix, f_posterior, targets):
        pass


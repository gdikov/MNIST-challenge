import scipy.sparse.linalg as spsl
import scipy.linalg as spl
import scipy.sparse as spm
import scipy.integrate as spi
import numpy as np

from numerics.softmax import softmax, logistic
from utils.label_factory import OneHotLabels, BinaryLabels

n_classes = 10


class LaplaceApproximation(object):
    def __init__(self):
        self.step_size = 1e-2
        self.max_iter = 20
        self.epsilon = 1e-5
        self.old_value = np.inf


    def approximate_binary(self, cov_matrix, targets, latent_init, cls):
        print("\t\tComputing the Laplace approximation with Newton iterations")
        self.binary_targets = BinaryLabels(class_one=cls).generate_labels(targets)

        self.iter_counter = 0
        self.num_samples = targets.shape[0]

        # initialise the temporary result storage variables and the latent function
        f = latent_init.copy()

        # for more details see Algorithm 3.1, p46 from Rasmussen's Gaussian Processes
        while (not self._is_converged(f)):
            pi = logistic(f)
            # W = - delta^2 log p(y|f)
            neg_hessian = pi - pi ** 2
            neg_hessian_sqrt = spm.diags(np.sqrt(neg_hessian), format='csc')
            # L = cholesky(I + W^(1/2) * K * W^(1/2))
            L = spl.cholesky((spm.identity(self.num_samples) +
                              neg_hessian_sqrt.dot(spm.csc_matrix(cov_matrix).dot(neg_hessian_sqrt)))
                             .toarray(), lower=True)
            # b = W * f + delta log p(y|f)
            b = spm.diags(neg_hessian, format='csc').dot(f) + self.binary_targets - pi
            # a = b - W^(1/2) * L^T \ (L \ (W^(1/2) * K * b))
            a = b - neg_hessian_sqrt.dot(spsl.spsolve(spm.csc_matrix(L.T),
                                                      spsl.spsolve(spm.csc_matrix(L),
                                                                   neg_hessian_sqrt.dot(cov_matrix.dot(b)))))
            # f = K * a
            f = cov_matrix.dot(a)

        approx_log_marg_likelihood = -0.5 * a.dot(f) + np.sum(np.log(logistic(f))) - np.sum(np.log(np.diagonal(L)))

        return f, a, approx_log_marg_likelihood


    def compute_latent_mean_cov_binary(self, cov_matrix_train, cov_matrix_test, f_posterior):
        num_test_samples = cov_matrix_test['hetero'].shape[0]
        pi = logistic(f_posterior)
        # W = - delta^2 log p(y|f_hat)
        neg_hessian = pi - pi ** 2
        neg_hessian_sqrt = spm.diags(np.sqrt(neg_hessian), format='csc')
        # L = cholesky(I + W^(1/2) * K * W^(1/2))
        L = spl.cholesky((spm.identity(self.num_samples) +
                          neg_hessian_sqrt.dot(spm.csc_matrix(cov_matrix_train).dot(neg_hessian_sqrt)))
                         .toarray(), lower=True)
        # mean_f* = k(x_*)^T * delta log p(y|f_hat)
        latent_means = cov_matrix_test['hetero'].dot(self.binary_targets - pi)
        # v = L \ (W^(1/2) * k(x_*))
        v = spsl.spsolve(spm.csc_matrix(L), neg_hessian_sqrt.dot(cov_matrix_test['hetero'].T)).T
        # V[f*] = k(x_*, x_*) - v^T * v
        latent_covs = cov_matrix_test['auto'] - np.einsum('ij,ij->i', v, v)

        return latent_means, latent_covs


    def approximate_posterior_integral(self, latent_mean, latent_cov):
        def approximate_integral_one_variable(_mean, _cov):
            _cov = np.abs(_cov)     # safety measure, it shouldn't happen but it happens...there is probably a bug
            const = 1.0 / (np.sqrt(2 * np.pi * _cov))
            neg_half_cov = -0.5 / _cov
            fun = lambda x: np.exp(neg_half_cov * (x - _mean)**2) / (1.0 + np.exp(-x))
            pi_star = const * spi.quad(fun, -10, 10)[0]
            if pi_star > 1 or pi_star < 0:
                print("ERRR")
                raise ValueError
            return pi_star

        pi_star_all = np.array([approximate_integral_one_variable(m, c) for m, c in zip(latent_mean, latent_cov)])
        return pi_star_all




    def approximate_multiclass(self, cov_matrix, targets, latent_init):
        print("\t\tComputing the Laplace approximation with Newton iterations")
        self.one_hot_targets = OneHotLabels(n_classes).generate_labels(targets)

        self.iter_counter = 0
        self.num_samples = targets.shape[0]

        # initialise the temporary result storage variables and the latent function
        f = latent_init.copy()

        # for more details see Algorithm 3.3, p50 from Rasmussen's Gaussian Processes
        while (not self._is_converged(f)):
            pi = softmax(f)
            E = list()
            z = list()
            for cls in xrange(n_classes):
                # compute pi and use it as Pi too
                pi_c = spm.diags(np.sqrt(pi[cls]), format='csc')
                # cholesky(I + D_c^(1/2) * K * D_c^(1/2))
                L = spl.cholesky((spm.identity(self.num_samples) +
                                  pi_c.dot(spm.csc_matrix(cov_matrix[cls]).dot(pi_c))).toarray(), lower=True)
                # E_c = D_c^(1/2) * L^T \ (L \ D_c^(1/2))
                E.append((pi_c.dot(spsl.spsolve(spm.csc_matrix(L.T), spsl.spsolve(spm.csc_matrix(L), pi_c))))
                         .toarray())
                # z_c = sum_i log(L_ii)
                z.append(np.sum(np.log(np.diagonal(L))))
            E = np.asarray(E)
            # M = cholesky(sum_c E_c)
            M = spl.cholesky(np.sum(E, axis=0), lower=True)
            b = list()
            c = list()
            for cls in xrange(n_classes):
                # compute Pi * Pi^T * f, Note that Pi * Pi^T is symmetric -> possible optimization!
                PiPiTf_cls = np.zeros(self.num_samples)
                for cls_prime in xrange(n_classes):
                    PiPiTf_cls += pi[cls] * pi[cls_prime] * f[cls_prime]
                # b = (D - Pi * Pi^T) * f + y - pi
                b_cls = pi[cls] * f[cls] - PiPiTf_cls + self.one_hot_targets[:, cls] - pi[cls]
                # c = E * K * b
                c_cls = E[cls].dot((cov_matrix[cls].dot(b_cls)))
                b.append(b_cls)
                c.append(c_cls)
            c = np.asarray(c)
            b = np.asarray(b)
            # a = b - c + E * R * M^T \ (M \ (R^T * c))
            a = (b.ravel() - c.ravel() + np.vstack(E).dot(spl.solve(M.T, spl.solve(M, np.sum(c, axis=0))))) \
                .reshape((n_classes, -1))
            # f = K * a
            for cls in xrange(n_classes):
                f[cls] = cov_matrix[cls].dot(a[cls])

        approx_log_marg_likelihood = -0.5 * a.ravel().dot(f.ravel()) \
                                     + self.one_hot_targets.T.ravel().dot(f.ravel()) \
                                     - np.sum(np.log(np.sum(np.exp(f), axis=0))) - np.sum(z)

        return f, approx_log_marg_likelihood


    def _is_converged(self, f):
        if self.iter_counter < self.max_iter:
            print("\t\t\tIteration {}: ||f|| = {}".format(self.iter_counter + 1, spl.norm(f)))
            self.iter_counter += 1
            # check the magnitude of the posterior or if the objective has increased in value
            if np.fabs(spl.norm(f) - self.old_value) < self.epsilon:
                self.old_value = np.inf
                return True
            self.old_value = spl.norm(f)
            return False
        print("\t\tNewton iterations are not converging and max iteration count has been attained")
        return True


    def compute_latent_mean_cov_multiclass(self, cov_matrix_train, cov_matrix_test, f_posterior):
        print("\t\tComputing latent mean and covariance")
        num_test_samples = cov_matrix_test['hetero'][0].shape[0]
        # for more details see Algorithm 3.4, p51 from Rasmussen's Gaussian Processes
        pi = softmax(f_posterior)
        E = list()
        z = list()
        for cls in xrange(n_classes):
            # compute pi and use it as Pi too
            pi_c = np.sqrt(pi[cls])
            # cholesky(I + D_c^(1/2) * K * D_c^(1/2))
            L = spl.cholesky((spm.identity(self.num_samples) +
                              spm.diags(pi_c).dot(spm.csc_matrix(cov_matrix_train[cls]).dot(spm.diags(pi_c))))
                             .toarray(), lower=True)
            # E_c = D_c^(1/2) * L^T \ (L \ D_c^(1/2))
            E.append((spsl.spsolve(spm.csc_matrix((L * pi_c).T),
                                   spsl.spsolve(spm.csc_matrix(L),
                                                spm.diags(pi_c, format='csc')))).toarray())
        E = np.asarray(E)
        # M = cholesky(sum_c E_c)
        M = spl.cholesky(np.sum(E, axis=0), lower=True)
        latent_means = np.zeros((num_test_samples, n_classes))
        latent_covs = np.zeros((num_test_samples, n_classes, n_classes))
        for cls in xrange(n_classes):
            latent_means[:, cls] = cov_matrix_test['hetero'][cls].dot(self.one_hot_targets[:, cls] - pi[cls])
            b = cov_matrix_test['hetero'][cls].dot(E[cls].T)
            c = E[cls].dot(spl.solve(M.T, spl.solve(M, b.T)))
            for cls_hat in xrange(n_classes):
                latent_covs[:, cls, cls_hat] = np.einsum('ij,ij->i', cov_matrix_test['hetero'][cls_hat], c.T)
            latent_covs[:, cls, cls] += cov_matrix_test['auto'][cls] - \
                                        np.einsum('ij,ij->i', cov_matrix_test['hetero'][cls], b)

        return latent_means, latent_covs

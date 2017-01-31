import scipy.sparse.linalg as spsl
import scipy.linalg as spl
import scipy.sparse as spm
import numpy as np

from numerics.softmax import softmax
from utils.label_factory import OneHot

n_classes = 10

class LaplaceApproximation():
    def __init__(self):
        self.step_size = 1e-2
        self.max_iter = 20
        self.epsilon = 1e-5
        self.old_value = np.inf


    def approximate(self, cov_matrix, targets, latent_init):
        print("\t\tComputing the Laplace approximation with Newton iterations")
        self.one_hot_targets = OneHot(n_classes).generate_labels(targets)

        self.iter_counter = 0
        self.num_samples = targets.shape[0]

        # initialise the temporary result storage variables and the latent function
        f = latent_init.copy()

        # for more details see Algorithm 3.3, p50 from Rasmussen's Gaussian Processes
        while(not self._is_converged(f)):
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
            a = (b.ravel() - c.ravel() + np.vstack(E).dot(spl.solve(M.T, spl.solve(M, np.sum(c, axis=0)))))\
                .reshape((n_classes, -1))
            # f = K * a
            for cls in xrange(n_classes):
                f[cls] = cov_matrix[cls].dot(a[cls])

            approx_log_marg_likelihood = -0.5 * a.ravel().dot(f.ravel()) \
                                         + self.one_hot_targets.T.ravel().dot(f.ravel()) \
                                         - np.sum(np.log(np.sum(np.exp(f), axis=0))) - np.sum(z)
            print(approx_log_marg_likelihood, -0.5 * a.ravel().dot(f.ravel()), self.one_hot_targets.T.ravel().dot(f.ravel())
                  - np.sum(np.log(np.sum(np.exp(f), axis=0))), np.sum(z))
        return f, approx_log_marg_likelihood


    def _is_converged(self, f):
        if self.iter_counter < self.max_iter:
            print("\t\t\tIteration {}: ||f|| = {}".format(self.iter_counter+1, spl.norm(f)))
            self.iter_counter += 1
            # check the magnitude of the posterior or if the objective has increased in value
            if np.fabs(spl.norm(f) - self.old_value) < self.epsilon:
                self.old_value = np.inf
                return True
            self.old_value = spl.norm(f)
            return False
        print("\t\tNewton iterations are not converging and max iteration count has been attained")
        return True


    def compute_latent_mean_cov(self, cov_matrix_train, cov_matrix_test, f_posterior):
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
                             .toarray())
            # E_c = D_c^(1/2) * L^T \ (L \ D_c^(1/2))
            E.append((spsl.spsolve(spm.csc_matrix((L * pi_c).T),
                                   spsl.spsolve(spm.csc_matrix(L),
                                                spm.diags(pi_c, format='csc')))).toarray())
        E = np.asarray(E)
        # M = cholesky(sum_c E_c)
        M = spl.cholesky(np.sum(E, axis=0))
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
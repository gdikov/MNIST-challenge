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
        self.max_iter = 1000


    def approximate(self, cov_matrix, targets):
        one_hot_targets = OneHot(n_classes).generate_labels(targets)
        self.iter_counter = 0
        self.num_samples = targets.shape[0]
        f = np.zeros((n_classes, self.num_samples))
        # for more details see Algorithm 3.3, p50 from Rasmussen's Gaussian Processes
        while(not self._is_converged()):
            pi = softmax(f)
            E = list()
            z = list()
            for cls in xrange(n_classes):
                # compute pi and use it as Pi too
                pi_c = np.sqrt(pi[cls])
                # cholesky(I + D_c^(1/2) * K * D_c^(1/2))
                L = spl.cholesky(spm.identity(self.num_samples) + (cov_matrix[cls].T * (pi_c**2)).T)
                # E_c = D_c^(1/2) * L^T \ (L \ D_c^(1/2))
                E.append(spsl.spsolve((L * pi_c).T, spsl.spsolve(L, spm.diags(pi_c))))
                # z_c = sum_i log(L_ii)
                z.append(np.log(np.prod(np.diagonal(L))))
            E = np.asarray(E)
            # M = cholesky(sum_c E_c)
            M = spl.cholesky(np.sum(E, axis=0))
            b = list()
            c = list()
            for cls in xrange(n_classes):
                # compute Pi * Pi^T * f, Note that Pi * Pi^T is symmetric -> if it slow exploit that structure!
                PiPif_cls = np.zeros(self.num_samples)
                for all_other_cls in xrange(n_classes):
                    PiPif_cls += pi[cls] * pi[all_other_cls] * f[all_other_cls]
                # b = (D - Pi * Pi^T) * f + y - pi
                b_cls = pi[cls] * f[cls] - PiPif_cls + targets[:, cls] - pi[cls]
                # c = E * K * b
                c_cls = E[cls].dot((cov_matrix[cls].dot(b_cls)))
                b.append(b_cls)
                c.append(c_cls)
            c = np.asarray(c)
            b = np.asarray(b)
            # a = b - c + E * R * M^T \ (M \ (R^T * c))
            a = (b.ravel() - c.ravel() + spl.solve(E.dot(M.T), spl.solve(M, np.sum(c, axis=0)))).reshape((n_classes, -1))
            # f = K * a
            for cls in xrange(n_classes):
                f[cls] = cov_matrix[cls].dot(a[cls])
        # return f_hat




    def _is_converged(self):
        if self.iter_counter < self.max_iter:
            self.iter_counter += 1
            # TODO: check the magnitude of the gradient or if the objective has increased in value
            return False
        return True

import numpy as np

from numerics.softmax import softmax

n_classes = 10

class AbstractSampler(object):
    def __init__(self, sampling_steps):
        self.sampling_steps = sampling_steps
        pass

class MonteCarlo(AbstractSampler):
    def __init__(self, num_sampling_steps):
        super(MonteCarlo, self).__init__(sampling_steps=num_sampling_steps)


    def sample(self, latent_mean, latent_cov):
        num_samples = latent_mean.shape[0]
        pi_star = np.zeros((num_samples, n_classes))

        for k in xrange(num_samples):
            f_sampled = np.random.multivariate_normal(mean=latent_mean[k], cov=latent_cov[k], size=self.sampling_steps)
            # class posterior is a softmax
            pi_star[k, :] = np.sum(softmax(f_sampled), axis=0)
        pi_star /= float(self.sampling_steps)

        return pi_star


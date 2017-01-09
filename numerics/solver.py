import numpy as np

class Solver(object):
    def __init__(self):
        pass

    def update(self, x, dx):
        pass


class SGD(Solver):
    def __init__(self, config=None):
        super(SGD, self).__init__()
        if config is None:
            config = dict()
            config['learning_rate'] = 0.01
            config['momentum'] = 0.9
            config['t'] = 0
        self.config = config


    def update(self, x, dx):
        x -= self.config['learning_rate'] * dx


class Adam(Solver):
    def __init__(self, data_dim=28*28, config=None):
        super(Adam, self).__init__()
        if config is None:
            config = dict()
            config['learning_rate'] = 0.0001
            config['beta1'] = 0.9
            config['beta2'] = 0.999
            config['epsilon'] = 1e-8
            config['mov_avg_grad'] = np.zeros(data_dim)
            config['mov_abg_sq_grad'] = np.zeros(data_dim)
            config['t'] = 0
        self.config = config


    def update(self, x, dx):
        self.config['mov_avg_grad'] = self.config['beta1'] * self.config['mov_avg_grad'] + (1 - self.config['beta1']) * dx
        self.config['mov_abg_sq_grad'] = self.config['beta2'] * self.config['mov_abg_sq_grad'] + (1 - self.config['beta2']) * (dx ** 2)
        self.config['t'] += 1
        m_hat = self.config['mov_avg_grad'] / (1 - self.config['beta1'] ** self.config['t'])
        v_hat = self.config['mov_abg_sq_grad'] / (1 - self.config['beta2'] ** self.config['t'])
        x -= (self.config['learning_rate'] * m_hat) / (np.sqrt(v_hat) + self.config['epsilon'])


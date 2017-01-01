from numerics.solver import Adam

class AbstractLayer(object):
    def __init__(self, incoming):
        self.incoming = incoming
        self.params = None
        self.dparams = None
        self.cache = None

    def output_shape(self):
        return None

    def init_params(self):
        pass

    def forward(self, **kwargs):
        return None

    def backward(self, upstream_derivatives):
        return None

    def update_trainable_params(self):
        if self.params is not None:
            for param in self.params.keys():
                self.solvers[param].update(self.params[param], self.dparams[param])

    def intit_solvers(self):
        if self.dparams is not None:
            self.solvers = {k: Adam(data_dim=self.dparams[k].shape) for k in self.dparams.keys() if k != 'X'}
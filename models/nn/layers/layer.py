

class AbstractLayer(object):
    def __init__(self, incoming):
        self.incoming = incoming
        self.params = None
        self.cache = None

    def output_shape(self):
        return None

    def init_params(self):
        pass

    def forward(self, **kwargs):
        return None

    def backward(self, upstream_derivatives):
        return None
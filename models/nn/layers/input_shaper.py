from models.nn.layers import AbstractLayer
import config as cfg

class Input(AbstractLayer):
    def __init__(self):
        super(Input, self).__init__(incoming=None)

    def init_params(self):
        pass

    def output_shape(self):
        return (cfg.batch_size, cfg.image_channels, cfg.image_height, cfg.image_height)

    def forward(self, raw_data):
        # reshape into the right dimensions
        return raw_data.reshape(cfg.batch_size, cfg.image_channels, cfg.image_height, cfg.image_height)

    def backward(self, upstream_derivatives):
        return upstream_derivatives
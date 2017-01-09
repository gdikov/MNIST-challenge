from models.nn.layers import AbstractLayer
import config as cfg

class Input(AbstractLayer):
    def __init__(self, mode='img'):
        super(Input, self).__init__(incoming=None)
        self.mode = mode

    def init_params(self):
        pass

    def output_shape(self):
        if self.mode == 'img':
            return (cfg.batch_size, cfg.image_channels, cfg.image_height, cfg.image_height)
        elif self.mode == 'vec':
            return (cfg.batch_size, cfg.image_channels * cfg.image_height * cfg.image_height)

    def forward(self, raw_data):
        # reshape into the right dimensions
        if self.mode == 'img':
            return raw_data.reshape(cfg.batch_size, cfg.image_channels, cfg.image_height, cfg.image_height)
        elif self.mode == 'vec':
            return raw_data.reshape(cfg.batch_size, cfg.image_channels * cfg.image_height * cfg.image_height)

    def backward(self, upstream_derivatives):
        return upstream_derivatives
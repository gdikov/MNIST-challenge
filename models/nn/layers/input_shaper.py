from models.nn.layers import AbstractLayer
import models.nn.config as cfg

class Input(AbstractLayer):
    def __init__(self, shape='img'):
        super(Input, self).__init__(incoming=None)
        self.shape_mode = shape

    def init_params(self):
        pass

    def output_shape(self):
        if self.shape_mode == 'img':
            return (cfg.batch_size, cfg.image_channels, cfg.image_height, cfg.image_height)
        elif self.shape_mode == 'vec':
            return (cfg.batch_size, cfg.image_channels * cfg.image_height * cfg.image_height)

    def forward(self, raw_data, mode='train'):
        # reshape into the right dimensions
        if self.shape_mode == 'img':
            return raw_data.reshape(cfg.batch_size, cfg.image_channels, cfg.image_height, cfg.image_height)
        elif self.shape_mode == 'vec':
            return raw_data.reshape(cfg.batch_size, cfg.image_channels * cfg.image_height * cfg.image_height)

    def backward(self, upstream_derivatives):
        return upstream_derivatives
import numpy as np

from models.model import AbstractModel


class ConvolutionalNeuralNetwork(AbstractModel):
    def __init__(self, batch_size=100):
        super(ConvolutionalNeuralNetwork, self).__init__('ConvNet')
        self.batch_size = batch_size
        self.build_network()
        self.initialise_parameters()


    def build_network(self):
        pass


    def fit(self, train_data, **kwargs):
        # reshape the input so that a channel dimension is added
        self.data = train_data
        # reshape the images into row vectors of 28*28 elements
        num_samples, dim_x, dim_y = self.data['x_train'].shape
        self.data['x_train'] = self.data['x_train'].reshape(num_samples, dim_x * dim_y)
        if self.add_bias:
            self.data['x_train'] = np.hstack((self.data['x_train'], np.ones((num_samples, 1))))
        # one_hot_labels = OneHot(10).generate_labels(self.data['y_train'])

        num_epochs = kwargs.get('num_epochs', 100)
        for i in xrange(num_epochs):
            losses = []
            for idx in self._batch_idx():
                loss, dW = softmax_loss(self.W,
                                        self.data['x_train'][idx],
                                        self.data['y_train'][idx],
                                        reg=self.regularisation)
                self.solver.update(self.W, dW)
                losses.append(loss)
            print("Epoch: {0}, loss: {1}".format(i, np.mean(losses)))
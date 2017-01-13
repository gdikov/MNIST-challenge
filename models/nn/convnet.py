from models.model import AbstractModel
from models.nn.layers import *

from numerics.softmax import softmax

import config as cfg
import os
import cPickle

import time

from utils.vizualiser import plot_filters


class ConvolutionalNeuralNetwork(AbstractModel):
    def __init__(self):
        super(ConvolutionalNeuralNetwork, self).__init__('ConvNet')
        self.batch_size = cfg.batch_size
        self._build_network()
        self.train_history = {'train_loss': [],
                              'val_acc': []}


    def _build_network(self):
        """
        Build a modified version of LeNet
        :return:
        """
        inp_layer = Input()

        conv1 = Conv(incoming=inp_layer,
                     conv_params={'stride': 1, 'pad': (5 - 1) / 2, 'filter_size': 5},
                     num_filters=20)
        relu1 = ReLU(incoming=conv1)
        pool1 = Pool(incoming=relu1,
                     pool_params={'pool_height': 2, 'pool_width': 2, 'stride': 2})

        conv2 = Conv(incoming=pool1,
                     conv_params={'stride': 1, 'pad': (5 - 1) / 2, 'filter_size': 5},
                     num_filters=50)
        relu2 = ReLU(incoming=conv2)
        pool2 = Pool(incoming=relu2, pool_params={'pool_height': 2, 'pool_width': 2, 'stride': 2})

        linear1 = Linear(incoming=pool2, num_units=500)
        lrelu1 = ReLU(incoming=linear1)

        dropout1 = Dropout(incoming=lrelu1, p=0.5)

        out_layer = Linear(incoming=dropout1, num_units=10)
        self.layers = (inp_layer,
                       conv1, relu1, pool1,
                       conv2, relu2, pool2,
                       linear1, lrelu1,
                       out_layer)


    def save_trainable_params(self):
        path_to_params = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pretrained')
        if not os.path.exists(path_to_params):
            os.makedirs(path_to_params)
        for layer_id, layer in enumerate(self.layers):
            if layer.params is not None:
                with open(os.path.join(path_to_params, 'layer_{0}.npy'.format(layer_id)), 'wb') as f:
                    cPickle.dump(layer.params, f)


    def load_trainable_params(self):
        path_to_params = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pretrained')
        if not os.path.exists(path_to_params):
            os.makedirs(path_to_params)
            print("Path to pre-trained parameters not found at: {}".format(path_to_params))
            raise IOError
        for layer_id, layer in enumerate(self.layers):
            if layer.params is not None:
                print("Loading params for layer {0}".format(layer_id))
                with open(os.path.join(path_to_params, 'layer_{0}.npy'.format(layer_id)), 'rb') as f:
                    layer.params = cPickle.load(f)


    def _compute_forward_pass(self, inp_data_batch, mode):
        out_data_batch = self.layers[0].forward(inp_data_batch)
        for layer_id in xrange(1, len(self.layers)):
            out_data_batch = self.layers[layer_id].forward(out_data_batch, mode=mode)
        return out_data_batch


    def _compute_backward_pass(self, end_derivatives):
        # update the last layer manually
        upstream_derivatives = self.layers[-1].backward(end_derivatives)
        self.layers[-1].update_trainable_params()
        for layer_id in xrange(len(self.layers)-2, 0, -1):
            upstream_derivatives = self.layers[layer_id].backward(upstream_derivatives)
            self.layers[layer_id].update_trainable_params()
        return upstream_derivatives


    def _compute_loss(self, scores, targets):
        num_train = scores.shape[0]
        probabilities = softmax(scores)
        loss = -np.sum(np.log(probabilities[np.arange(num_train), targets])) / num_train
        probabilities[np.arange(num_train), targets] -= 1
        dsoftmax = probabilities / num_train
        return loss, dsoftmax


    def _batch_idx(self, data_size, shuffle=True):
        if shuffle:
            # maybe this is unnecessary because they are already shuffled
            # but it doesn't harm much to do it again
            shuffled_order = np.random.permutation(np.arange(data_size))
        else:
            shuffled_order = np.arange(data_size)
        for x in np.array_split(shuffled_order, data_size // self.batch_size):
            yield x


    def fit(self, train_data, **kwargs):
        # reshape the input so that a channel dimension is added
        self.data = train_data
        # reshape the images into row vectors of 28*28 elements
        num_samples, dim_x, dim_y = self.data['x_train'].shape
        self.data['x_train'] = self.data['x_train'].reshape(num_samples, dim_x * dim_y)

        num_epochs = kwargs.get('num_epochs', 100)
        best_val_acc = 0.0
        for i in xrange(num_epochs):
            epoch_losses = []
            for idx in self._batch_idx(num_samples):
                scores = self._compute_forward_pass(self.data['x_train'][idx], mode='train')
                loss, dscores = self._compute_loss(scores, self.data['y_train'][idx])
                self._compute_backward_pass(dscores)
                self.train_history['train_loss'].append(loss)
                epoch_losses.append(loss)
                print("Minibatch train loss: {}".format(loss))
            # validate
            val_predictions = self.predict(data['x_val'])
            val_acc = np.sum(val_predictions == data['y_val']) / float(val_predictions.shape[0]) * 100.
            self.train_history['val_acc'].append(val_acc)

            if val_acc > best_val_acc:
                print("Saving weights")
                self.save_trainable_params()
                best_val_acc = val_acc
            print("Epoch: {0}, mean loss: {1}, validation accuracy: {2}".format(i, np.mean(epoch_losses), val_acc))


    def predict(self, new_data):
        # reshape the input so that a channel dimension is added
        # reshape the images into row vectors of 28*28 elements
        num_samples, dim_x, dim_y = new_data.shape
        new_data = new_data.reshape(num_samples, dim_x * dim_y)

        scores_all = []
        for i, idx in enumerate(self._batch_idx(num_samples, shuffle=False)):
            scores = self._compute_forward_pass(new_data[idx], mode='test')
            scores_all.append(scores)
        scores_all = np.concatenate(scores_all)
        return np.argmax(scores_all, axis=1)


if __name__ == "__main__":
    from utils.data_utils import load_MNIST

    data = load_MNIST()

    model = ConvolutionalNeuralNetwork()

    # model.load_trainable_params()
    # plot_filters(model.layers[1].params['W'], plot_shape=(2,10), channel=1)
    model.fit(data, num_epochs=100)
    #
    # predictions = model.predict(data['x_test'][:1000])
    # #
    # test_acc = np.sum(predictions == data['y_test'][:1000]) / float(predictions.shape[0]) * 100.
    # print("Validation accuracy: {0}"
    #       .format(test_acc))
    #
    # miscalssified_idx = predictions != data['y_val'][:100]
    # from utils.vizualiser import plot_digits
    # #
    # plot_digits(data['x_val'][:100][miscalssified_idx][:64], predictions[miscalssified_idx][:64], plot_shape=(8, 8))
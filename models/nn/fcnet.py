from models.model import AbstractModel
from models.nn.layers import *

from numerics.softmax import softmax

import config as cfg
import os
import cPickle


class BasicNeuralNetwork(AbstractModel):
    def __init__(self):
        super(BasicNeuralNetwork, self).__init__('BasicNet')
        self.batch_size = cfg.batch_size
        self._build_network()


    def _build_network(self):
        """
        Build a modified version of LeNet
        :return:
        """
        inp_layer = Input(shape='vec')

        linear1 = Linear(incoming=inp_layer, num_units=200)
        lrelu1 = ReLU(incoming=linear1)
        out_layer = Linear(incoming=lrelu1, num_units=10)
        self.layers = (inp_layer,
                       linear1, lrelu1,
                       out_layer)


    def save_trainable_params(self):
        path_to_params = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pretrained_basicnet')
        if not os.path.exists(path_to_params):
            os.makedirs(path_to_params)
        for layer_id, layer in enumerate(self.layers):
            if layer.params is not None:
                with open(os.path.join(path_to_params, 'layer_{0}.npy'.format(layer_id)), 'wb') as f:
                    cPickle.dump(layer.params, f)


    def load_trainable_params(self):
        path_to_params = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pretrained_basicnet')
        if not os.path.exists(path_to_params):
            os.makedirs(path_to_params)
            print("Path to pre-trained parameters not found at: {}".format(path_to_params))
            raise IOError
        for layer_id, layer in enumerate(self.layers):
            if layer.params is not None:
                print("Loading params for layer {0}".format(layer_id))
                with open(os.path.join(path_to_params, 'layer_{0}.npy'.format(layer_id)), 'rb') as f:
                    layer.params = cPickle.load(f)


    def _compute_forward_pass(self, inp_data_batch):
        out_data_batch = self.layers[0].forward(inp_data_batch)
        for layer_id in xrange(1, len(self.layers)):
            out_data_batch = self.layers[layer_id].forward(out_data_batch)
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
        best_epoch_loss = np.infty
        for i in xrange(num_epochs):
            losses = []
            for idx in self._batch_idx(num_samples):
                scores = self._compute_forward_pass(self.data['x_train'][idx])
                loss, dscores = self._compute_loss(scores, self.data['y_train'][idx])
                self._compute_backward_pass(dscores)
                losses.append(loss)
            mean_loss = np.mean(losses)
            # if mean_loss < best_epoch_loss:
            #     print("Saving weights on epoch: {0}".format(i))
            #     self.save_trainable_params()
            #     best_epoch_loss = mean_loss
            print("Epoch: {0}, loss: {1}".format(i, mean_loss))


    def predict(self, new_data):
        # reshape the input so that a channel dimension is added
        # reshape the images into row vectors of 28*28 elements
        num_samples, dim_x, dim_y = new_data.shape
        new_data = new_data.reshape(num_samples, dim_x * dim_y)

        scores_all = []
        for idx in self._batch_idx(num_samples, shuffle=False):
            scores = self._compute_forward_pass(new_data[idx])
            scores_all.append(scores)
            print("{0}".format(np.mean(scores)))
        scores_all = np.concatenate(scores_all)
        return np.argmax(scores_all, axis=1)


if __name__ == "__main__":
    from utils.data_utils import load_MNIST

    data = load_MNIST()

    model = BasicNeuralNetwork()

    # model.load_trainable_params()

    model.fit(data, num_epochs=30)

    predictions = model.predict(data['x_val'])
    #
    test_acc = np.sum(predictions == data['y_val']) / float(predictions.shape[0]) * 100.
    print("Validation accuracy: {0}"
          .format(test_acc))

    miscalssified_idx = predictions != data['y_val']
    from utils.vizualiser import plot_digits

    plot_digits(data['x_val'][miscalssified_idx][:64], predictions[miscalssified_idx][:64], plot_shape=(8, 8))
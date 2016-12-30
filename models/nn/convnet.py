

from models.model import AbstractModel
from models.nn.layers import *

from numerics.solver import SGD
from numerics.softmax import softmax

import config as cfg


class ConvolutionalNeuralNetwork(AbstractModel):
    def __init__(self, batch_size=100):
        super(ConvolutionalNeuralNetwork, self).__init__('ConvNet')
        self.batch_size = cfg.batch_size
        self.solver = SGD()
        self._build_network()


    def _build_network(self):
        inp_layer = Input()
        filter_size = 3
        conv_params = {'stride': 1, 'pad': (filter_size - 1) / 2}
        fst_conv = Conv(incoming=inp_layer, conv_params=conv_params, num_filters=5)
        fst_relu = ReLU(incoming=fst_conv)
        pool_params = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        fst_pool = Pool(incoming=fst_relu, pool_params=pool_params)
        snd_linear = Linear(incoming=fst_pool, num_units=11)
        snd_relu = ReLU(incoming=snd_linear)
        out_layer = Linear(incoming=snd_relu, num_units=10)
        self.layers = (inp_layer, fst_conv, fst_relu, fst_pool, snd_linear, snd_relu, out_layer)


    def _compute_forward_pass(self, inp_data_batch):
        out_data_batch = self.layers[0].forward(inp_data_batch)
        for layer_id in xrange(1, len(self.layers)):
            out_data_batch = self.layers[layer_id].forward(out_data_batch)
        return out_data_batch


    def _compute_backward_pass(self, end_derivatives):
        def update_trainable_params_for(layer):
            for param in layer.dparams.keys():
                if not param == 'X':
                    self.solver.update(layer.params[param], layer.dparams[param])

        # update the last layer manually
        upstream_derivatives = self.layers[-1].backward(end_derivatives)
        update_trainable_params_for(self.layers[-1])
        for layer_id in xrange(len(self.layers)-2, 0, -1):
            upstream_derivatives = self.layers[layer_id].backward(upstream_derivatives)
            update_trainable_params_for(self.layers[layer_id])
        return upstream_derivatives


    def _compute_loss(self, scores, targets):
        num_train = scores.shape[0]
        probabilities = softmax(scores)
        loss = -np.sum(np.log(probabilities[np.arange(num_train), targets])) / num_train
        probabilities[np.arange(num_train), targets] -= 1
        dsoftmax = probabilities / num_train
        return loss, dsoftmax


    def _batch_idx(self):
        # maybe this is unneceserray because they are already shuffled
        # but it doesn't harm much to do it again
        num_training = self.data['x_train'].shape[0]
        shuffled_order = np.random.permutation(np.arange(num_training))
        for x in np.array_split(shuffled_order, num_training // self.batch_size):
            yield x


    def fit(self, train_data, **kwargs):
        # reshape the input so that a channel dimension is added
        self.data = train_data
        # reshape the images into row vectors of 28*28 elements
        num_samples, dim_x, dim_y = self.data['x_train'].shape
        self.data['x_train'] = self.data['x_train'].reshape(num_samples, dim_x * dim_y)

        num_epochs = kwargs.get('num_epochs', 100)

        for i in xrange(num_epochs):
            losses = []
            for idx in self._batch_idx():
                scores = self._compute_forward_pass(self.data['x_train'][idx])
                loss, dscores = self._compute_loss(scores, self.data['y_train'][idx])
                self._compute_backward_pass(dscores)
                losses.append(loss)
                print(loss)
            print("Epoch: {0}, loss: {1}".format(i, np.mean(losses)))


if __name__ == "__main__":
    from utils.data_utils import load_MNIST

    data = load_MNIST(num_training=50000, num_validation=10000)

    model = ConvolutionalNeuralNetwork(batch_size=500)

    model.fit(data, num_epochs=300)

    # predictions = model.predict(data['x_val'])

    # test_acc = np.sum(predictions == data['y_val']) / float(predictions.shape[0]) * 100.
    # print("Validation accuracy: {0}"
    #       .format(test_acc))
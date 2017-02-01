import numpy as np
import os

from models.model import AbstractModel
from numerics.solver import Adam
from numerics.softmax import softmax_loss, softmax

class LogisticRegression(AbstractModel):

    def __init__(self, batch_size=10000, add_bias=False):
        super(LogisticRegression, self).__init__('LogisticRegression')
        self.data = None
        self.batch_size = batch_size
        self.add_bias = add_bias
        self.num_classes = 10
        solver_config = {'learning_rate': 0.13, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8, 't': 0}
        if add_bias:
            solver_config['mov_avg_grad'] = np.zeros(28*28+1)
            solver_config['mov_avg_sq_grad'] = np.zeros(28*28+1)
        else:
            solver_config['mov_avg_grad'] = np.zeros(28*28)
            solver_config['mov_avg_sq_grad'] = np.zeros(28*28)
        self.solver = Adam(config=solver_config)
        self._init_params()


    def _init_params(self):
        if self.add_bias:
            self.W = 0.01 * np.random.randn(self.num_classes, 28*28)
            self.W = np.hstack((self.W, np.zeros((self.num_classes, 1))))
        else:
            self.W = 0.01 * np.random.randn(self.num_classes, 28 * 28)


    def save_trainable_params(self):
        path_to_params = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'optimal_W.npy')
        np.save(path_to_params, self.W)


    def load_trainable_params(self):
        path_to_params = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'optimal_W.npy')
        if not os.path.exists(path_to_params):
            print("\tPath to parameters not found at: {}".format(path_to_params))
            raise IOError
        self.W = np.load(path_to_params)


    def fit(self, train_data, **kwargs):

        self.data = train_data
        # reshape the images into row vectors of 28*28 elements
        num_samples, dim_x, dim_y = self.data['x_train'].shape
        self.data['x_train'] = self.data['x_train'].reshape(num_samples, dim_x * dim_y)
        if self.add_bias:
            self.data['x_train'] = np.hstack((self.data['x_train'], np.ones((num_samples, 1))))
        # one_hot_labels = OneHot(10).generate_labels(self.data['y_train'])

        num_epochs = kwargs.get('num_epochs', 100)
        regularisation = kwargs.get('reg', 0.0)
        reinit = kwargs.get('reinit', True)
        verbose = kwargs.get('verbose', False)
        save_best = kwargs.get('save_best', False)
        if reinit:
            self._init_params()

        lowest_loss = np.inf
        for i in xrange(num_epochs):
            losses = []
            for idx in self._batch_idx():
                # scores = np.dot(self.W, self.data['x_train'][idx])
                loss, dW = softmax_loss(self.W,
                                        self.data['x_train'][idx],
                                        self.data['y_train'][idx],
                                        reg=regularisation)
                self.solver.update(self.W, dW)
                losses.append(loss)
            mean_loss = np.mean(losses)
            if verbose:
                print("\t\tEpoch: {0}, loss: {1}".format(i, mean_loss))
            if save_best:
                if mean_loss < lowest_loss:
                    lowest_loss = mean_loss
                    self.save_trainable_params()


    def _batch_idx(self):
        # maybe this is unneceserray because they are already shuffled
        # but it doesn't harm much to do it again
        num_training = self.data['x_train'].shape[0]
        shuffled_order = np.random.permutation(np.arange(num_training))
        for x in np.array_split(shuffled_order, num_training // self.batch_size):
            yield x


    def predict(self, new_data, **kwargs):
        # make sure the new_data is shaped like the train data
        if new_data.shape[1] != 28 * 28 + 1 and new_data.shape[1] != 28 * 28:
            new_data = new_data.reshape(new_data.shape[0], 28*28)
        if self.add_bias and new_data.shape[1] != 28*28 + 1:
            new_data = np.hstack((new_data, np.ones((new_data.shape[0], 1))))

        scores = np.dot(new_data, self.W.T)
        probs = softmax(scores)     # this is unnecessary as the softmax function will not change the order
        return np.argmax(probs, axis=1)


if __name__ == "__main__":
    from utils.data_utils import load_MNIST
    data = load_MNIST(num_training=50000, num_validation=10000)

    model = LogisticRegression(batch_size=50000, regularisation=0.0, add_bias=True)

    model.fit(data, num_epochs=300)

    predictions = model.predict(data['x_val'])

    test_acc = np.sum(predictions == data['y_val']) / float(predictions.shape[0]) * 100.
    print("Validation accuracy: {0}"
          .format(test_acc))

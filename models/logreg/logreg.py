import numpy as np

from models.model import AbstractModel
from numerics.solver import Adam
from numerics.softmax import softmax_loss, softmax
from utils.label_factory import OneHot

class LogisticRegression(AbstractModel):

    def __init__(self, batch_size=10000, regularisation=0.1, add_bias=False):
        super(LogisticRegression, self).__init__('LogisticRegression')
        self.data = None
        self.batch_size = batch_size
        self.regularisation = regularisation
        self.add_bias = add_bias
        self.num_classes = 10
        if add_bias:
            self.solver = Adam(data_dim=28*28+1)
        else:
            self.solver = Adam(data_dim=28*28)
        self._init_params()



    def _init_params(self):
        if self.add_bias:
            self.W = 0.01 * np.random.randn(self.num_classes, 28*28)
            self.W = np.hstack((self.W, np.zeros((self.num_classes, 1))))
        else:
            self.W = 0.01 * np.random.randn(self.num_classes, 28 * 28)


    def fit(self, train_data, **kwargs):

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


    def _batch_idx(self):
        # maybe this is unneceserray because they are already shuffled
        # but it doesn't harm much to do it again
        num_training = self.data['x_train'].shape[0]
        shuffled_order = np.random.permutation(np.arange(num_training))
        for x in np.array_split(shuffled_order, num_training // self.batch_size):
            yield x


    def predict(self, new_data):
        if self.add_bias:
            # make sure the new_data is shaped like the train data
            if new_data.shape[1] != self.data['x_train'].shape[1]:
                new_data = new_data.reshape(new_data.shape[0], self.data['x_train'].shape[1] - 1)
            new_data = np.hstack((new_data, np.ones((new_data.shape[0], 1))))
        else:
            if new_data.shape[1] != self.data['x_train'].shape[1]:
                new_data = new_data.reshape(new_data.shape[0], self.data['x_train'].shape[1])

        probs = softmax(self.W, new_data)
        return np.argmax(probs, axis=1)


if __name__ == "__main__":
    from utils.data_utils import load_MNIST
    data = load_MNIST(num_training=50000, num_validation=10000)

    model = LogisticRegression(batch_size=50000, regularisation=0.00, add_bias=True)

    model.fit(data, num_epochs=300)

    predictions = model.predict(data['x_val'])

    test_acc = np.sum(predictions == data['y_val']) / float(predictions.shape[0]) * 100.
    print("Validation accuracy: {0}"
          .format(test_acc))

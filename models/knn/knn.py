import numpy as np

from models.model import AbstractModel
from utils.hyper_opt import KFoldCrossValidation

class kNearestNeighbours(AbstractModel):

    def __init__(self, batch_size=10):
        super(kNearestNeighbours, self).__init__('kNN')
        self.data = None
        self.num_chunks = batch_size
        pass


    def fit(self, train_data, **kwargs):
        self.data = train_data
        # reshape the images into row vectors of 28*28 elements
        num_samples, dim_x, dim_y = self.data['x_train'].shape
        self.data['x_train'] = self.data['x_train'].reshape(num_samples, dim_x * dim_y)


    def predict(self, new_data, **kwargs):
        """
        Compute  a matrix of distances between unseen test data samples and the known training samples.
        Then predict a label for each test sample.

        @:param new_data: Nx28x28 matrix of unlabeled MNIST images
        @:param k: the number of the neighbours to vote for the label

        @:return: the predicted labels for the unseen data samples
        :param **kwargs:
        """

        # make sure the new_data is shaped like the train data
        if new_data.shape[1] != self.data['x_train'].shape[1]:
            new_data = new_data.reshape(new_data.shape[0], self.data['x_train'].shape[1])

        num_test = new_data.shape[0]

        y_pred = np.zeros(num_test)

        for chunk, idx in zip(np.array_split(new_data, self.num_chunks),
                              np.array_split(np.arange(new_data.shape[0]), self.num_chunks)):
            partial_distance_matrix = self._compute_distance_matrix(chunk)
            for local_id, global_id in enumerate(idx):
                closest_k = np.argpartition(partial_distance_matrix[local_id], k)[:k]
                labels = self.data['y_train'][closest_k]

                votes = np.bincount(labels)
                majority = np.argmax(votes)
                y_pred[global_id] = majority

        return y_pred


    def _compute_distance_matrix(self, new_data, dist_mode='L2'):
        """
        Compute the distance between each test and training sample

        @:param new_data: the unseen test/validation data samples
        @:param dist_mode: is the distance measure between the training and test/validation samples

        @:return the distance matrix
        """

        if dist_mode == 'L2':
            distances_chunk = np.sum(new_data ** 2, axis=1, keepdims=True) \
                              + np.sum(self.data['x_train'] ** 2, axis=1) \
                              - 2 * np.dot(new_data, self.data['x_train'].T)

            return distances_chunk
        else:
            raise NotImplementedError



if __name__ == "__main__":
    from utils.data_utils import load_MNIST
    data = load_MNIST(num_training=60000, num_validation=0)

    model = kNearestNeighbours()

    validator = KFoldCrossValidation(data=data, k=3)
    best_k = validator.validate(model, xrange(1, 10))

    model.fit(data)
    predictions = model.predict(data['x_test'], k=best_k)

    test_acc = np.sum(predictions == data['y_test']) / float(predictions.shape[0]) * 100.
    print("Test accuracy for k={1}: {0}"
          .format(test_acc, best_k))

    miscalssified_idx = predictions != data['y_test']

    from utils.vizualiser import plot_digits

    plot_digits(data['x_test'][miscalssified_idx][:64], predictions[miscalssified_idx][:64], plot_shape=(8, 8))

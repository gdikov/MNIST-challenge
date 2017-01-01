import numpy as np
from models.model import AbstractModel


class GaussianProcesses(AbstractModel):
    def __init__(self):
        super(GaussianProcesses, self).__init__('GaussianProcesses')


    def fit(self, train_data, **kwargs):
        pass

    def predict(self, new_data):
        return None


if __name__ == "__main__":
    from utils.data_utils import load_MNIST

    data = load_MNIST(num_training=50000, num_validation=1000)

    model = GaussianProcesses()

    model.fit(data)

    predictions = model.predict(data['x_val'])

    test_acc = np.sum(predictions == data['y_val']) / float(predictions.shape[0]) * 100.
    print("Validation accuracy: {0}"
          .format(test_acc))
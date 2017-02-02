import os
import numpy as np

from models.knn.knn import kNearestNeighbours
from models.logreg.logreg import LogisticRegression
from models.nn.convnet import ConvolutionalNeuralNetwork


from utils.hyper_opt import KFoldCrossValidation

from utils.data_utils import load_MNIST

path_to_models = os.path.join(os.path.realpath(os.path.dirname(__file__)), '../models')
path_to_results = os.path.join(os.path.realpath(os.path.dirname(__file__)), '../report')


def evaluate_knn(train_from_scratch=False, verbose=True):
    """
    Test the kNN classifier on the whole test set using the whole training and validation set
    :return: mean test accuracy (i.e. percentage of the correctly classified samples)
    """
    data_train, data_test = load_MNIST(num_training=60000, num_validation=0)

    model = kNearestNeighbours()

    path_to_optimal = os.path.join(path_to_models, 'knn/optimal_k.npy')
    if not train_from_scratch and os.path.exists(path_to_optimal):
        best_k = np.load(path_to_optimal)
        print("\tLoading pre-computed optimal parameter k={}".format(best_k))
    else:
        validator = KFoldCrossValidation(data=data_train, k=5)
        best_k = validator.validate(model=model, ranges=xrange(1, 10), verbose=verbose)
        np.save(path_to_optimal, best_k)

    model.fit(data_train)
    predictions = model.predict(data_test['x_test'], k=best_k)

    test_acc = np.sum(predictions == data_test['y_test']) / float(predictions.shape[0]) * 100.

    # log the result from the test
    np.save(os.path.join(path_to_results, 'predictions_knn.npy'), predictions)

    del data_train, data_test, model
    return test_acc


def evaluate_logreg(train_from_scratch=False, verbose=True):
    """
    Test the Logistic Regression classifier on the whole testing set using the a subsets for  training and validation.
    :return: mean test accuracy (i.e. percentage of the correctly classified samples)
    """
    data_train, data_test = load_MNIST(num_training=60000, num_validation=0)

    model = LogisticRegression(batch_size=10000, add_bias=True)

    if train_from_scratch or not os.path.exists(os.path.join(path_to_models, 'logreg/optimal_W.npy')):
        validator = KFoldCrossValidation(data=data_train, k=5)
        regularisation_range = [0., 1e-5, 1e-4, 1e-3, 1e-2]
        best_reg = validator.validate(model=model, ranges=regularisation_range, verbose=verbose)
        model.fit(data_train, num_epochs=100, reg=best_reg, reinit=True, verbose=True, save_best=True)

    print("\tLoading pre-computed optimal weight matrix W")
    model.load_trainable_params()

    predictions = model.predict(data_test['x_test'])

    test_acc = np.sum(predictions == data_test['y_test']) / float(predictions.shape[0]) * 100.

    # log the result from the test
    np.save(os.path.join(path_to_results, 'predictions_logreg.npy'), predictions)

    del data_train, data_test, model
    return test_acc


def evaluate_convnet(train_from_scratch=True, verbose=True):
    """
    Test the Convolutional Neural Network classifier on the whole testing set
    using the a subsets for training and validation.
    :return: mean test accuracy (i.e. percentage of the correctly classified samples)
    """
    from utils.data_utils import load_MNIST

    data_train, data_test = load_MNIST()

    model = ConvolutionalNeuralNetwork()

    exist_pretrained = os.path.exists(os.path.join(path_to_models, 'nn/pretrained/layer_1.npy')) and \
                       os.path.exists(os.path.join(path_to_models, 'nn/pretrained/layer_4.npy')) and \
                       os.path.exists(os.path.join(path_to_models, 'nn/pretrained/layer_7.npy')) and \
                       os.path.exists(os.path.join(path_to_models, 'nn/pretrained/layer_10.npy'))

    if train_from_scratch or not exist_pretrained:
        answ = raw_input("\tTraining from scratch can take some days on a notebook. "
                         "Do you want to load the pre-computed weights instead? [yes]/no")
        if not answ.startswith('y'):
            model.fit(data_train, num_epochs=20)

    model.load_trainable_params()
    predictions = model.predict(data_test['x_test'])

    test_acc = np.sum(predictions == data_test['y_test']) / float(predictions.shape[0]) * 100.

    # log the result from the test
    np.save(os.path.join(path_to_results, 'predictions_convnet.npy'), predictions)

    del data_train, data_test, model
    return test_acc

def test_gp():
    pass
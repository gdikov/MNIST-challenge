import numpy as np

def softmax(scores):
    """
    Computes only the softmax activation

    @:param W: the weights for all classes of shape (num_classes, D)
    @:param X: the data samples of shape (num_samples, D)

    @:return probabilities for each sample of belonging to each class
    """
    log_c = -np.max(scores, axis=1)
    scores_better_numeric = np.exp((scores.T + log_c).T)
    sum_scores = np.sum(scores_better_numeric, axis=1)
    probabilities = (scores_better_numeric.T / sum_scores).T

    return probabilities


def softmax_loss(W, x, y, reg=0):
    """
    Computes the softmax loss function.

    @:param W: the weights for all classes of shape (num_classes, D)
    @:param X: the data samples of shape (num_samples, D)
    @:param y: the output from the softmax of shape (num_samples, num_classes)

    @:returns loss and gradient of X w.r.t W
    """

    num_train = x.shape[0]
    scores = np.dot(x, W.T)
    probabilities = softmax(scores)

    loss = -np.sum(np.log(probabilities[np.arange(num_train), y])) / num_train
    loss += 0.5 * reg * np.sum(W * W)

    probabilities[np.arange(num_train), y] -= 1
    dW = np.dot(probabilities.T, x) / num_train
    dW += reg * W

    return loss, dW
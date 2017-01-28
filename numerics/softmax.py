import numpy as np

def softmax(scores):
    """
    Computes only the softmax activation
    :param scores is a NxD matrix of N scores for D classes
    :return a NxD matrix with the probabilities for each class for each sample
    """
    log_c = -np.max(scores, axis=1)
    scores_better_numeric = np.exp((scores.T + log_c).T)
    sum_scores = np.sum(scores_better_numeric, axis=1)
    probabilities = (scores_better_numeric.T / sum_scores).T

    return probabilities


def softmax_loss(W, x, y, reg=0):
    """
    Computes the softmax loss function.
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
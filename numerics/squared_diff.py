import numpy as np



def squared_loss(W, x, y, reg=0.1):
    """
    Computes the softmax loss function.

    @:param W: the weights for all classes of shape (num_classes, D)
    @:param X: the data samples of shape (num_samples, D)
    @:param y: the output from the softmax of shape (num_samples, num_classes)

    @:returns loss and gradient of X w.r.t W
    """

    num_train = x.shape[0]

    loss = 0.5 * np.sum((np.argmax(np.dot(x, W.T)) - y) ** 2) / num_train
    loss += 0.5 * reg * np.sum(W * W)

    dW = np.zeros_like(W)# TODO np.dot(probabilities.T, x) / num_train
    dW += reg * W

    return loss, dW
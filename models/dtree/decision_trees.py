import numpy as np
import os

# TODO: adapt the code for the MNIST dataset, inheriting from abstract model

class DecisionTree():
    def __init__(self, num_classes, id="root"):
        self.id = id
        self.is_terminal = True

        self.class_dist = np.zeros(num_classes)
        self.p_dist = np.zeros(num_classes)
        self.gini_index = 1.

        self.decision_feature_id = 0
        self.decision_threshold = 0.

        self.lq = None  # Left DT: less than or equal to the threshold
        self.gt = None  # Right DT: greater than the threshold

        """ Compute the new class and probability distribution as well as the impurity (Gini index) """

    def _update_node(self, ys):
        self.class_dist += np.bincount(ys, minlength=self.class_dist.shape[0])
        self.p_dist = self.class_dist / np.sum(self.class_dist)
        self.gini_index = 1 - np.sum(self.p_dist ** 2)

        """ Compute the infomation gain for all possible splittings within a feature and all possible features """

    def _compute_igain(self, xs, ys):
        num_examples, num_features = xs.shape
        igain_max = 0.
        e_max = f_id_max = 0
        datax_lq = datay_lq = datax_gt = datay_gt = None
        if self.is_terminal and self.gini_index > 0.:
            for f in xrange(num_features):
                for e in xrange(num_examples):
                    all_lq = (xs[:, f] <= xs[e, f] - 1e-8)
                    num_all_lq = all_lq.sum()
                    p_lq = num_all_lq / float(num_examples)
                    p_gt = 1 - p_lq

                    datax_lq_tmp = xs[all_lq]
                    datay_lq_tmp = ys[all_lq]
                    datax_gt_tmp = xs[~all_lq]
                    datay_gt_tmp = ys[~all_lq]

                    gini_lq = 1 - np.sum((np.bincount(datay_lq_tmp) / float(num_all_lq)) ** 2)
                    gini_gt = 1 - np.sum((np.bincount(datay_gt_tmp) / float(num_examples - num_all_lq)) ** 2)

                    if self.gini_index - p_lq * gini_lq - p_gt * gini_gt > igain_max:
                        igain_max = self.gini_index - p_lq * gini_lq - p_gt * gini_gt
                        datax_lq = datax_lq_tmp
                        datay_lq = datay_lq_tmp
                        datax_gt = datax_gt_tmp
                        datay_gt = datay_gt_tmp
                        e_max = e
                        f_id_max = f

        return igain_max, e_max, f_id_max, datax_lq, datay_lq, datax_gt, datay_gt

        """ Internal trainig method according to the depth and information gain criteria """

    def _train_on(self, dataset, depth):
        xs, ys = dataset
        # update root parameters
        self._update_node(ys)

        if depth > 0:
            igain_max, e_max, f_id_max, datax_lq, datay_lq, datax_gt, datay_gt = self._compute_igain(xs, ys)
            if igain_max > 0.:
                # perform split
                self.is_terminal = False
                self.decision_threshold = xs[e_max, f_id_max] - 1e-8
                self.decision_feature_id = f_id_max
                self.lq = DecisionTree(num_classes=self.class_dist.shape[0], id="lq_{0}".format(depth))
                self.gt = DecisionTree(num_classes=self.class_dist.shape[0], id="gt_{0}".format(depth))

                self.lq.train((datax_lq, datay_lq), depth=depth - 1)
                self.gt.train((datax_gt, datay_gt), depth=depth - 1)

        """ Public train method which also estimates the accuracy """

    def train(self, dataset, depth=2):
        xs, ys = dataset
        # train on the whole dataset
        self._train_on(dataset, depth=depth)

        # check the accuracy and error
        acc = 0
        for ind, x in enumerate(xs):
            y, _ = self.test(x)
            if y == ys[ind]:
                acc += 1
        return float(acc) * 100. / xs.shape[0]

        """ Test the decision tree with some unseen example """

    def test(self, x):
        if self.is_terminal:
            return np.argmax(self.class_dist), self.p_dist[np.argmax(self.class_dist)]
        else:
            if x[self.decision_feature_id] <= self.decision_threshold:
                return self.lq.test(x)
            return self.gt.test(x)

        """ Print attributes of a node """

    def print_node(self):
        print("-------------------------------------")
        if self.is_terminal:
            print("- Terminal Node ID: {0}".format(self.id))
        else:
            print("- Node ID: {0}".format(self.id))
        print("- Class distribution: {0}".format(self.class_dist))
        print("- Probability distribution: {0}".format(self.p_dist))
        print("- Impurity (Gini index): {0}".format(self.gini_index))
        if not self.is_terminal:
            print("- Decision feature id: {0}".format(self.decision_feature_id))
            print("- Decision threshold: {0}".format(self.decision_threshold))
        print("-------------------------------------")

        """ Print the whole tree from the top (root) to bottom (terminal nodes/leafs)"""

    def print_tree(self):
        if self.is_terminal:
            self.print_node()
        else:
            self.print_node()
            self.lq.print_tree()
            self.gt.print_tree()

    """ Load the dataset from a file as numpy arrays """


def load_dataset(filename):
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    train_data = np.genfromtxt(filepath, delimiter=', ')
    # strip the row with the labels
    train_data = train_data[1:, :]
    # decouple the labels
    labels = train_data[:, -1]
    labels = labels.astype(np.int64)
    # strip the labels from the features
    train_data = train_data[:, :-1]

    return train_data, labels

    """ Inspect the dataset in 3d plots """


def plot_dataset(xs, ys):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(xs[:, 0], xs[:, 1], xs[:, 2], s=25, marker='s', lw=0, c=ys)
    ax.scatter([4.1], [-0.1], [2.2], s=25, marker='*')
    ax.scatter([6.1], [0.4], [1.3], s=25, marker='*')

    ax.set_xlabel('0')
    ax.set_ylabel('1')
    ax.set_zlabel('2')

    plt.show()



if __name__ == '__main__':
    train_data, labels = load_dataset("sample_dataset.csv")

    dt = DecisionTree(num_classes=np.max(labels) + 1)

    accuracy = dt.train(dataset=(train_data, labels), depth=2)
    print("Final training accuracy: {0}%".format(accuracy))
    dt.print_tree()
    x_a = np.array([4.1, -0.1, 2.2])
    x_b = np.array([6.1, 0.4, 1.3])

    y_a, p_a = dt.test(x_a)
    y_b, p_b = dt.test(x_b)

    print("x_a was classified as {0} with p(c=y_a | x_a, T) = {1}".format(y_a, p_a))
    print("x_b was classified as {0} with p(c=y_b | x_b, T) = {1}".format(y_b, p_b))

    # plot_dataset(train_data, labels)


import numpy as np
import os

from models.model import AbstractModel

n_classes = 3

class DecisionTree(AbstractModel):
    def __init__(self):
        super(DecisionTree, self).__init__('DecisionTree')
        self.dtree = _RecursiveTree(num_classes=n_classes)

    def fit(self, train_data, **kwargs):
        depth = kwargs.get('depth', 1)
        # fit on the whole dataset
        self.dtree.fit(train_data, depth=depth)

    def predict(self, new_data, **kwargs):
        predictions = np.zeros(new_data.shape[0])
        for i, x in enumerate(new_data):
            predictions[i] = self.dtree.predict(x)
        return predictions


class _RecursiveTree():
    def __init__(self, num_classes, id_="root"):
        self.id = id_
        self.is_terminal = True

        self.class_dist = np.zeros(num_classes)
        self.p_dist = np.zeros(num_classes)
        self.gini_index = 1.

        self.decision_feature_id = 0
        self.decision_threshold = 0.

        self.lq = None  # Left DT: less than or equal to the threshold
        self.gt = None  # Right DT: greater than the threshold


    def _update_node(self, ys):
        self.class_dist += np.bincount(ys, minlength=self.class_dist.shape[0])
        self.p_dist = self.class_dist / np.sum(self.class_dist)
        self.gini_index = 1 - np.sum(self.p_dist ** 2)


    def _compute_igain(self, xs, ys):
        num_examples, num_features = xs.shape
        igain_max = 0.
        e_max = f_id_max = 0
        datax_lq = datay_lq = datax_gt = datay_gt = None
        if self.is_terminal and self.gini_index > 0.:
            for f in xrange(num_features):
                for e in xrange(num_examples):
                    # TODO: refactor the epsilon and compute boundary as middle point of two data points
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


    def fit(self, dataset, depth):
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
                self.lq = _RecursiveTree(num_classes=self.class_dist.shape[0], id_="lq_{0}".format(depth))
                self.gt = _RecursiveTree(num_classes=self.class_dist.shape[0], id_="gt_{0}".format(depth))

                self.lq.fit((datax_lq, datay_lq), depth=depth - 1)
                self.gt.fit((datax_gt, datay_gt), depth=depth - 1)


    def predict(self, x, return_probs=False):
        if self.is_terminal:
            if return_probs:
                return np.argmax(self.class_dist), self.p_dist[np.argmax(self.class_dist)]
            else:
                return np.argmax(self.class_dist)
        else:
            if x[self.decision_feature_id] <= self.decision_threshold:
                return self.lq.predict(x)
            return self.gt.predict(x)



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


    def print_tree(self):
        if self.is_terminal:
            self.print_node()
        else:
            self.print_node()
            self.lq.print_tree()
            self.gt.print_tree()


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

    return np.array(train_data), np.array(labels)



if __name__ == '__main__':
    train_data = load_dataset("sample_dataset.csv")

    model = DecisionTree()
    depth = 2

    model.fit(train_data, depth=depth)
    predictions = model.predict(train_data[0])

    test_acc = np.sum(predictions == train_data[1]) / float(predictions.shape[0]) * 100.
    print("Test accuracy for depth={1}: {0}"
          .format(test_acc, depth))



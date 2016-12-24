import numpy as np

class CrossValidation(object):
    def __init__(self, data):
        if isinstance(data, dict):
            if 'x_val' in data.keys() and 'x_train' in data.keys():
                data['x'] = np.concatenate((data['x_train'], data['x_val']))
                del data['x_train'], data['x_val']
                data['y'] = np.concatenate((data['y_train'], data['y_val']))
                del data['y_train'], data['y_val']
        self.trainval = data

    def validate(self, model, ranges, verbose):
        return None


class KFoldCrossValidation(CrossValidation):

    def __init__(self, data, k):
        super(KFoldCrossValidation, self).__init__(data)
        self.k = k

    def validate(self, model, ranges, verbose=True):
        print("Performing k-fold cross-validation with k={}".format(self.k))
        # TODO: refactor this to be a real k-fold cross-validation
        if model.name == 'kNN':
            if isinstance(ranges, xrange):
                param_range = ranges
            else:
                raise ValueError("ranges must be a xrange-instance")

            accs = []
            for k in param_range:
                micro_acc = []
                for train_data, val_data in self.get_folds():
                    model.fit(train_data)
                    predictions = model.predict(val_data['x'], k=k)
                    macc = np.sum(predictions == val_data['y']) / float(predictions.shape[0]) * 100.0
                    micro_acc.append(macc)
                averaged_acc_for_k = np.mean(micro_acc)
                if verbose:
                    print("Validation accuracy for k={1}: {0}".format(averaged_acc_for_k, k))
                accs.append(averaged_acc_for_k)
            best_k = np.argmax(accs) + 1
            print("Best k is: {0}".format(best_k))
            return best_k
        else:
            raise NotImplementedError


    def get_folds(self):
        data_size = self.trainval['x'].shape[0]
        folds = np.array_split(np.arange(data_size), self.k)
        # if the last split is unbalanced, merge with the one before the last
        if folds[self.k - 1].shape[0] < folds[0].shape[0] // 2:
            folds[self.k - 2] = np.concatenate([folds[self.k - 2], folds[self.k - 1]])
            del folds[self.k - 1]

        for i in xrange(self.k):
            val_idxs = folds[i]
            train_idxs = np.setdiff1d(np.arange(data_size), val_idxs, assume_unique=True)
            train_data = {k: self.trainval[k][train_idxs] for k in ['x', 'y']}
            val_data = {k: self.trainval[k][val_idxs] for k in ['x', 'y']}
            yield train_data, val_data



class MonteCarloCrossValidation(CrossValidation):
    def __init__(self, data, k):
        super(MonteCarloCrossValidation, self).__init__(data)
        self.k = k

    def validate(self, model, ranges, verbose):
        print("Performing Monte Carlo cross-validation with k={}".format(self.k))
        if model.name == 'kNN':
            if isinstance(ranges, xrange):
                param_range = ranges
            else:
                raise ValueError("ranges must be a xrange-instance")
            accs = []
            for k in param_range:
                predictions = model.predict(self.trainval['x'], k=k)
                acc = np.sum(predictions == self.trainval['y']) / float(predictions.shape[0]) * 100.0
                if verbose:
                    print("Validation accuracy for k={1}: {0}".format(acc, k))
                accs.append(acc)
            best_k = np.argmax(accs) + 1
            print("Best k is: {0}".format(best_k))
            return best_k
        else:
            raise NotImplementedError

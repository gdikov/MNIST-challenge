import numpy as np
import cPickle
import os

class CrossValidation(object):
    def __init__(self, data):
        self.trainval = data

    def validate(self, model, ranges, verbose):
        return None


class KFoldCrossValidation(CrossValidation):

    def __init__(self, data, k):
        super(KFoldCrossValidation, self).__init__(data)
        self.k = k

    def validate(self, model, ranges, verbose=True):
        print("\tPerforming {}-fold cross-validation".format(self.k))
        if model.name == 'kNN':
            if isinstance(ranges, xrange):
                param_range = ranges
            else:
                raise ValueError("ranges must be an xrange instance")

            accs = []
            for k in param_range:
                micro_acc = []
                for train_data, val_data in self._get_folds():
                    model.fit(train_data)
                    predictions = model.predict(val_data['x'], k=k)
                    macc = np.sum(predictions == val_data['y']) / float(predictions.shape[0]) * 100.0
                    micro_acc.append(macc)
                averaged_acc_for_k = np.mean(micro_acc)
                if verbose:
                    print("\t\tValidation accuracy for k={1}: {0}".format(averaged_acc_for_k, k))
                accs.append(averaged_acc_for_k)
            best_k = np.argmax(accs) + 1
            print("\tBest k is: {0}".format(best_k))
            return best_k

        elif model.name == 'LogisticRegression':
            if isinstance(ranges, list):
                param_range = ranges
            else:
                raise ValueError("ranges must be a list instance")

            accs = []
            for reg in param_range:
                micro_acc = []
                for train_data, val_data in self._get_folds():
                    model.fit(train_data, num_epochs=50, reg=reg, reinit=True, verbose=False, save_best=False)
                    predictions = model.predict(val_data['x'])
                    macc = np.sum(predictions == val_data['y']) / float(predictions.shape[0]) * 100.0
                    micro_acc.append(macc)
                averaged_acc_for_epoch = np.mean(micro_acc)
                if verbose:
                    print("\t\tValidation accuracy for reg={1}: {0}".format(averaged_acc_for_epoch, reg))
                accs.append(averaged_acc_for_epoch)

            best_reg = np.argmax(accs)
            print("\tBest reg is: {0}".format(param_range[best_reg]))
            return param_range[best_reg]

        else:
            raise NotImplementedError


    def _get_folds(self):
        data_size = self.trainval['x_train'].shape[0]
        folds = np.array_split(np.arange(data_size), self.k)
        # if the last split is unbalanced, merge with the one before the last
        if folds[self.k - 1].shape[0] < folds[0].shape[0] // 2:
            folds[self.k - 2] = np.concatenate([folds[self.k - 2], folds[self.k - 1]])
            del folds[self.k - 1]

        for i in xrange(self.k):
            val_idxs = folds[i]
            train_idxs = np.setdiff1d(np.arange(data_size), val_idxs, assume_unique=True)
            train_data = {k: self.trainval[k][train_idxs] for k in ['x_train', 'y_train']}
            val_data = {k: self.trainval[k + '_train'][val_idxs] for k in ['x', 'y']}
            yield train_data, val_data



# class MonteCarloCrossValidation(CrossValidation):
#     def __init__(self, data, k):
#         super(MonteCarloCrossValidation, self).__init__(data)
#         self.k = k
#
#     def validate(self, model, ranges, verbose):
#         print("Performing Monte Carlo cross-validation with k={}".format(self.k))
#         if model.name == 'kNN':
#             if isinstance(ranges, xrange):
#                 param_range = ranges
#             else:
#                 raise ValueError("ranges must be a xrange-instance")
#             accs = []
#             for k in param_range:
#                 predictions = model.predict(self.trainval['x_train'], k=k)
#                 acc = np.sum(predictions == self.trainval['y_train']) / float(predictions.shape[0]) * 100.0
#                 if verbose:
#                     print("Validation accuracy for k={1}: {0}".format(acc, k))
#                 accs.append(acc)
#             best_k = np.argmax(accs) + 1
#             print("Best k is: {0}".format(best_k))
#             return best_k
#         else:
#             raise NotImplementedError


class AbstractModel(object):
    def __init__(self, name='model'):
        self.name = name

    def fit(self, train_data, **kwargs):
        pass

    def predict(self, new_data, **kwargs):
        return None

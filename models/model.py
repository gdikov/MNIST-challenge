
class AbstractModel(object):
    def __init__(self, name='model'):
        self.name = name

    def fit(self, train_data):
        pass

    def predict(self, new_data, mode='val'):
        return None

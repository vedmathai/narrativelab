from factuality.common.config import Config


class AbstractDataHandler():
    def __init__(self):
        super().__init__()
        self._config = Config.instance()

    def load(self):
        pass

    def labels(self):
        labels = set()
        for datum in self.data().data():
            if isinstance(datum.label(), list):
                for label in datum.label():
                    labels.add(label)
            else:
                labels.add(datum.label())
        return labels
    
    def label2idx(self):
        labels = self.labels()
        label2idx = {}
        for label_i, label in enumerate(labels):
            label2idx[label] = label_i
        idx2label = {v: k for k, v in label2idx.items()}
        return label2idx, idx2label
    
    def train_data(self):
        data = self.data().data()
        return data[:int(0.7 * len(data))]
    
    def evaluate_data(self):
        data = self.data().data()
        return data[int(0.7 * len(data)) : int(0.8 * len(data))]
    
    def test_data(self):
        data = self.data().data()
        return data[int(0.8 * len(data)):]

    def to_dict(self):
        return {
            'data': self.data(),
        }
from factuality.data.nela.nela_model.nela_labels import NelaLabels
from factuality.data.nela.nela_model.nela_data import NelaData


class NelaDataset:
    def __init__(self):
        self._labels = None
        self._data = []

    def labels(self):
        return self._labels
    
    def data(self):
        return self._data
    
    def set_labels(self, labels):
        self._labels = labels

    def set_data(self, data):
        self._data = data

    def add_data(self, data):
        self._data.append(data)

    @staticmethod
    def from_dict_and_csv(data_dicts, label_val):
        nela_dataset = NelaDataset()
        labels = NelaLabels.from_csv(label_val)
        nela_dataset.set_labels(labels)
        for d in data_dicts:
            data = NelaData.from_dict(d, data_dicts[d], labels)
            nela_dataset.add_data(data)
        return nela_dataset
        
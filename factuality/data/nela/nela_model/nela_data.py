from Levenshtein import distance
import json


from factuality.data.nela.nela_model.nela_datum import NelaDatum
from factuality.data.nela.nela_model.nela_file2data_mapping import nela_file2data_mapping


class NelaData:
    def __init__(self):
        self._name = None
        self._data = []
        self._label = None

    def name(self):
        return self._name
    
    def data(self):
        return self._data
    
    def label(self):
        return self._label
    
    def set_name(self, name):
        self._name = name

    def set_data(self, data):
        self._data = data

    def add_datum(self, datum):
        self._data.append(datum)

    def set_label(self, label):
        self._label = label

    @staticmethod
    def from_dict(name, data_val, labels):
        nela_data = NelaData()
        for item in data_val:
            datum = NelaDatum.from_dict(item)
            nela_data.add_datum(datum)
        min_distance = 1000
        s1 = ''.join(name[:-5].lower().split())
        if s1 in nela_file2data_mapping:
            required_label = nela_file2data_mapping[s1]
            for label in labels.labels():
                s2 = ''.join(label.name().lower().split())
                if s2 == required_label:
                    nela_data.set_label(label)
        nela_data.set_name(name[:-5])
        return nela_data

    def to_dict(self):
        return {
            "name": self.name(),
            "data": [i.to_dict() for i in self.data()],
            "label": self.label().to_dict() if self.label() is not None else None,
        }
    
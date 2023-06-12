from factuality.data.nela.nela_model.nela_label import NelaLabel


class NelaLabels:
    def __init__(self):
        self._labels = []

    def labels(self):
        return self._labels
    
    def set_labels(self, labels):
        self._labels = labels

    def add_label(self, label):
        self._labels.append(label)

    @staticmethod
    def from_csv(val):
        nela_labels = NelaLabels()
        for i in val:
            nela_label = NelaLabel.from_csv(i)
            nela_labels.add_label(nela_label)
        return nela_labels
    
    def to_dict(self):
        return [
            i.to_dict() for i in self.labels()
        ]
    
    @staticmethod
    def from_dict(labels):
        nela_labels = NelaLabels()
        nela_labels.set_name(labels['name'])
        nela_labels.set_media_bias_fact_check_label(labels['media_bias_fact_check_label'])
        return nela_labels
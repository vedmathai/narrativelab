from factuality.data.vitamin_c.readers.vitamin_c_model.vitamin_c_datum import VitaminCDatum


class VitaminCDataset:
    def __init__(self):
        self._data = []

    def data(self):
        return self._data

    def set_data(self, data):
        self._data = data

    @staticmethod
    def from_dict(val):
        dataset = VitaminCDataset()
        passage_data = [VitaminCDatum.from_dict(i) for i in val]
        dataset.set_data(passage_data)
        return dataset
    
    def to_dict(self):
        return [
            i.to_dict() for i in self.data()
        ]

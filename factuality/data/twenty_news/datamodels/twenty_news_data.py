class TwentyNewsData:
    def __init__(self):
        self._data = []

    def add_datum(self, datum):
        self._data.append(datum)

    def data(self):
        return self._data

    def add_datum(self, datum):
        self._data.append(datum)

    def set_data(self, data):
        self._data = data

    def to_dict(self):
        return {
            'data': self.data(),
        }

    def to_dict(self):
        return {
            'data': self.data(),
        }

    @staticmethod
    def from_dict(data_dict):
        eurlex_data = TwentyNewsData()
        eurlex_data.set_data(data_dict['_data'])
        return eurlex_data

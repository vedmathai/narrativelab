from factuality.tasks.quotes.datamodels.quote_datum import QuoteDatum


class QuoteData:
    def __init__(self):
        self._data = []

    def add_datum(self, datum):
        return self._data.append(datum)
    
    def data(self):
        return self._data
    
    def set_data(self, data):
        self._data = data

    @staticmethod
    def from_dict(val):
        quote_data = QuoteData()
        quote_data.set_data(QuoteDatum.from_dict(i) for i in val['quote_data'])
        return quote_data
    
    def to_dict(self):
        return {
            'quote_data': [i.to_dict() for i in self.data()]
        }

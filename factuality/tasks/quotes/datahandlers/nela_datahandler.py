import random
from collections import defaultdict

from factuality.common.config import Config
from factuality.tasks.quotes.datamodels.quote_datum import QuoteDatum
from factuality.tasks.quotes.datamodels.quote_data import QuoteData
from factuality.tasks.quotes.datareaders.nela_quotes_datareader import NelaQuotesDatareader

label_mapping = {
    'left_center_bias': 'left',
    'right_center_bias': 'right',
    'left_bias': 'left',
    'right_bias': 'right',
}

class NelaDatahandler():
    def __init__(self):
        self._data = None
        self._data_reader = NelaQuotesDatareader()

    def load(self):
        self._data_reader.load()
        self._load_data()
        
    def train_data(self):
        return self._train_data
    
    def test_data(self):
        return self._test_data
    
    def _load_data(self):
        data = self._data_reader.read_data()
        quote_data = QuoteData()
        for item in data:
            datum = QuoteDatum()
            datum.set_label(item[1])
            datum.set_text(item[0])
            quote_data.add_datum(datum)
        self._train_data, self._test_data = self._balance(quote_data.data())
        return quote_data
    
    def _balance(self, quote_data):
        random.seed(23)
        random.shuffle(quote_data)
        label2data = defaultdict(list)
        seen = set()
        train_data = QuoteData()
        test_data = QuoteData()
        for ri, r in enumerate(quote_data):
            if r.label()[:-4] not in label_mapping:
                continue
            label = label_mapping[r.label()[:-4]] + r.label()[-4:]
            r.set_label(label)
            label2data[label].append(r)
        train_doc_list = []
        test_doc_list = []
        min_size = min([len(i) for i in label2data.values()] + [5000])
        train_size = int(0.8 * min_size)
        for k in label2data:
            for i in label2data[k][:train_size]:
                train_data.add_datum(i)
        for k in label2data:
            for i in label2data[k][train_size:min_size]:
                test_data.add_datum(i)
        return train_data, test_data


if __name__ == '__main__':
    config = Config.instance()
    datahandler = NelaDatahandler()
    datahandler.load()

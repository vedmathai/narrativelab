import random
from collections import defaultdict

from factuality.common.config import Config
from factuality.tasks.agreement.datamodels.agreement_datum import AgreementDatum
from factuality.tasks.agreement.datamodels.agreement_data import AgreementData
from factuality.tasks.agreement.datareaders.agreement_datareader import AgreementDatareader


class AgreementDatahandler():
    def __init__(self):
        self._data = None
        self._data_reader = AgreementDatareader()

    def load(self):
        self._data_reader.load()
        self._load_data()
        
    def train_data(self):
        return self._train_data
    
    def test_data(self):
        return self._test_data
    
    def _load_data(self):
        data = self._data_reader.read_data()
        agreement_data = AgreementData()
        for item in data[1:]:
            datum = AgreementDatum()
            datum.set_label(item[0])
            datum.set_text_1(item[4])
            datum.set_text_2(item[5])
            agreement_data.add_datum(datum)
        self._train_data, self._test_data = self._balance(agreement_data.data())
        return agreement_data
    
    def _balance(self, quote_data):
        random.seed(23)
        random.shuffle(quote_data)
        label2data = defaultdict(list)
        seen = set()
        train_data = AgreementData()
        test_data = AgreementData()
        for ri, r in enumerate(quote_data):
            label = r.label()
            label2data[label].append(r)
        min_size = min([len(i) for i in label2data.values()] + [3000])
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
    datahandler = AgreementDatahandler()
    datahandler.load()

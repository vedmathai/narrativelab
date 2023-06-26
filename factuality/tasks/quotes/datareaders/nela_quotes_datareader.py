import csv

from factuality.common.config import Config


class NelaQuotesDatareader():
    def __init__(self):
        self._config = Config.instance()

    def load(self):
        pass

    def read_data(self):
        data = []
        with open(self._config.nela_quotes_data_location()) as f:
            reader = csv.reader(f, delimiter='\t')
            data = list(reader)
        return data

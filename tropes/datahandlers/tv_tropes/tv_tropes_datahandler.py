import csv
import os

from tropes.common.config.config import Config


class TVTropesDataHandler():
    # write init function
    def __init__(self):
        self._config = Config.instance()
        self._data = []
        self._labels = set()
        
    def load(self):
        data = []
        good_reads_data = os.path.join(self._config.tv_tropes_data_location(), 'lit_goodreads_match.csv')
        with open(good_reads_data, 'rt') as f:
            csv_reader = csv.reader(f, delimiter=',')
            next(csv_reader)  # skip the first row
            for row in csv_reader:
                self._data.append(row)
        labels = os.path.join(self._config.tv_tropes_data_location(), 'tropes.csv')
        with open(labels, 'rt') as f:
            csv_reader = csv.reader(f, delimiter=',')
            next(csv_reader)
            for row in csv_reader:
                self._labels.add(row[2])

    def data(self):
        return self._data

    def labels(self):
        return self._labels

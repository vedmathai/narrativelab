import csv
import json
import os

from factuality.data.nela.nela_model.nela_dataset import NelaDataset


class NelaDatareader():
    def __init__(self):
        self._label_location = "/home/lalady6977/oerc/projects/data/nela-elections-2020.json/labels.csv"
        self._data_location = "/home/lalady6977/oerc/projects/data/nela-elections-2020.json"

    def read_dataset(self):
        dataset = NelaDataset()
        with open(self._label_location) as f:
            reader = csv.reader(f, delimiter=',')
            labels = list(reader)
        news_data_location = os.path.join(self._data_location, 'nela-elections-2020/newsdata')
        filenames = os.listdir(news_data_location)
        data_dicts = {}
        for filename in filenames:
            filepath = os.path.join(news_data_location, filename)
            with open(filepath) as f:
                data_dicts[filename] = json.load(f)
        nela_dataset = NelaDataset.from_dict_and_csv(data_dicts, labels)
        return nela_dataset

if __name__ == '__main__':
    nela_datareader = NelaDatareader()
    dataset = nela_datareader.read_dataset()
    for i in dataset.data():
        print(i.to_dict())

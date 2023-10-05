import numpy as np
import random
from collections import defaultdict

from tropes.datahandlers.tv_tropes.tv_tropes_datahandler import TVTropesDataHandler
from tropes.common.config.config import Config
from tropes.entry_points.registries.datahandler_registry import DataHandlerRegistry
from tropes.trainer.trope_classification.datahandlers.model_input.model_input_data import ModelInputData
from tropes.trainer.trope_classification.datahandlers.model_input.model_input_datum import ModelInputDatum


class RoBERTaDataHandler():
    def __init__(self):
        self._config = Config.instance()
        self._data_handler_registry = DataHandlerRegistry.instance()
        self._data_handler = TVTropesDataHandler()
        self._model_input_data = ModelInputData()
        self._labels = self._data_handler.labels()
        self._train_data = []
        self._test_data = []
        self._validation_data = []
        self._data = []
        self._counts = defaultdict(int)

    def load(self):
        self._data_handler.load()
        self._model_input_data = ModelInputData()
        self.process_data()
        self.load_train_data()
        self.load_test_data()
        self.load_validation_data()

    def process_data(self):
        data_dict = defaultdict(list)
        data = []
        for datum in self._data_handler.data():
            data_dict[datum[2]].append(datum)
        for key in data_dict.keys():
            self._counts[len(data_dict[key])] += 1
            if len(data_dict[key]) > 200:
                print(key, len(data_dict[key]))
                data.extend(random.choices(data_dict[key], k=5000))
        random.shuffle(data)
        self._data = data

    def load_train_data(self):
        length = len(self._data)
        self._train_data = self._data[:int(length*0.6)]
        for datum in self._train_data:
            if datum[2] not in self._labels:
                continue
            model_input_datum = ModelInputDatum()
            model_input_datum.set_sentence(datum[3])
            model_input_datum.set_target(datum[2])
            self._model_input_data.add_train_datum(model_input_datum)
            self._model_input_data.add_class(model_input_datum.target())
    
    # A function that loops over valuation data and instantiates a model input datum for each
    def load_validation_data(self):
        length = len(self._data)
        self._validation_data = self._data[int(length*0.6):int(length*0.8)]
        for datum in self._validation_data:
            if datum[2] not in self._labels:
                continue
            model_input_datum = ModelInputDatum()
            model_input_datum.set_sentence(datum[3])
            model_input_datum.set_target(datum[2])
            self._model_input_data.add_validation_datum(model_input_datum)
            self._model_input_data.add_class(model_input_datum.target())

    # A function that loops over test data and instantiates a model input datum for each
    def load_test_data(self):
        length = len(self._data)
        self._test_data = self._data[int(length*0.8):]
        for datum in self._test_data:
            if datum[2] not in self._labels:
                continue
            model_input_datum = ModelInputDatum()
            model_input_datum.set_sentence(datum[3])
            model_input_datum.set_target(datum[2])
            self._model_input_data.add_test_datum(model_input_datum)
            self._model_input_data.add_class(model_input_datum.target())
    
    def model_input_data(self):
        return self._model_input_data

    def label_counts(self):
        return len(self._data_handler.labels())
    
    def datum_from_sentence(self, sentence):
        datum = ModelInputDatum()
        datum.set_sentence(sentence)
        return datum

    def label_map(self):
        labels = sorted(list(self._data_handler.labels()))
        label_to_index = {}
        index_to_label = {}
        for i, label in enumerate(labels):
            label_to_index[label] = i
            index_to_label[i] = label
        return label_to_index, index_to_label

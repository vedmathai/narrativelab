import numpy as np
import os
from torch import nn
import torch
from tqdm import tqdm
from torch.optim import Adam

from tropes.common.config.config import Config
from tropes.trainer.trope_classification.datahandlers.roberta_datahandler import RoBERTaDataHandler
from tropes.trainer.trope_classification.models.trope_classifier_model import TropeClassifierModel


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TropeClassificationInfer:
    def __init__(self):
        self._data_handler = RoBERTaDataHandler()
        self._config = Config.instance()

    def load(self, run_config={}):
        self._data_handler.load()
        self._label_to_index, self._index_to_label = self._data_handler.label_map()
        num_labels = self._data_handler.label_counts()
        self._model = TropeClassifierModel(run_config, num_labels)
        self._model.load()

    def classify(self, datum):
        output = self._model(datum)
        return output

    def infer(self, sentence):
        with torch.no_grad():
            datum = self._data_handler.datum_from_sentence(sentence)
            event_predicted_vector = self.classify(datum)
            predicted = event_predicted_vector.argmax(dim=1).item()
            predicted_label = self._index_to_label[predicted]
            return predicted_label


if __name__ == '__main__':
    tci = TropeClassificationInfer()
    tci.load()
    label = tci.infer('George went back to school')
    print(label)

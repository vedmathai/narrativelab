import numpy as np
import os
from torch import nn
import torch
from tqdm import tqdm
from torch.optim import Adam
from jadelogs import JadeLogger


from coreference.model.coreference_model import CoreferenceClassifierModel  # noqa
from coreference.datahandler.ecb.ecb_datahandler import ECBDatahandler


TRAIN_SAMPLE_SIZE = int(8000 / 5)
TEST_SAMPLE_SIZE = 2000
EPOCHS = 40
LEARNING_RATE = 1e-5  # 1e-2
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_EVERY = 10000000


class CoreferenceClassificationTrain:
    def __init__(self):
        self._jade_logger = JadeLogger()
        self._total_loss = 0
        self._all_losses = []
        self._iteration = 0
        self._last_iteration = 0
        self._loss = None

    def load(self, run_config):
        data_handler = 'coreference_datahandler'
        self._data_handler = ECBDatahandler()
        self._data_handler.load()
        self._model = CoreferenceClassifierModel(run_config)
        self._model_optimizer = Adam(
            self._model.parameters(),
            lr=LEARNING_RATE,
        )
        self._criterion = nn.CrossEntropyLoss()

    def zero_grad(self):
        self._model.zero_grad()

    def optimizer_step(self):
        self._model_optimizer.step()

    def train_step(self, datum):
        event_predicted_vector = self.classify(datum)
        relationship_target = self.relationship_target(datum)
        event_prediction_loss = self._criterion(
            event_predicted_vector, relationship_target
        )
        predicted = event_predicted_vector[0]
        if predicted > 0.5:
            predicted_label = 1
        else:
            predicted_label = 0
        if self._loss is None:
            self._loss = event_prediction_loss
        else:
            self._loss += event_prediction_loss
        return event_prediction_loss, predicted_label

    def relationship_target(self, datum):
        target = datum.label()
        relationship_target = torch.from_numpy(np.array([float(target), float(1-target)])).to(device)
        return relationship_target

    def classify(self, datum):
        output = self._model(datum)
        return output

    def train_epoch(self):
        self.zero_grad()
        train_sample = self._data_handler.train_data()
        self._jade_logger.new_train_batch()
        for datum in tqdm(train_sample):
            loss, predicted_label = self.train_step(datum)
            self._all_losses += [loss.item()]
            self._iteration += 1
            if self._loss is not None and self._iteration % 10 == 0:
                self._loss.backward()
                self.optimizer_step()
                self.zero_grad()
                self._loss = None
            self._jade_logger.new_train_datapoint(datum.label(), predicted_label, loss.item(), {})
            

    def train(self, run_config):
        self._jade_logger.new_experiment()
        self._jade_logger.set_experiment_type('classification')
        self._jade_logger.set_total_epochs(run_config.epochs())
        for epoch in range(EPOCHS):
            self._jade_logger.new_epoch()
            self._epoch = epoch
            self.train_epoch()
            self.evaluate(run_config)
            self._model.save()


    def evaluate(self, run_config):
        with torch.no_grad():
            test_sample = self._data_handler.test_data()
            self._jade_logger.new_evaluate_batch()
            for datumi, datum in enumerate(test_sample):
                event_predicted_vector = self.classify(datum)
                relationship_target = self.relationship_target(datum)
                batch_loss = self._criterion(
                    event_predicted_vector,
                    relationship_target
                )
                loss = batch_loss.item()
                predicted = event_predicted_vector[0]
                if predicted > 0.5:
                    predicted_label = 1
                else:
                    predicted_label = 0
                self._jade_logger.new_evaluate_datapoint(datum.label(), predicted_label, loss, {})

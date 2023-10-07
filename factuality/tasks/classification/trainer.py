import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from collections import defaultdict
from jadelogs import JadeLogger
import random


from factuality.common.config import Config
from factuality.tasks.classification.datamodels.classification_datum import ClassificationDatum
from factuality.tasks.classification.datahandlers.datahandlers_registry import DatahandlersRegistry
from factuality.tasks.classification.models.registry import ModelsRegistry


LEARNING_RATE = 1e-5
BATCH_SIZE = 24
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


sentence_breaks = {
    'bigbird': '[SEP]',
    'roberta': '</s>',
}

token_delimiters = {
    'bigbird': '▁',
    'roberta': 'Ġ',
}


class ClassificationTrainBase:
    def __init__(self):
        self._jade_logger = JadeLogger()
        self._datahandlers_registry = DatahandlersRegistry()
        self._models_registry = ModelsRegistry()

    def load(self, run_config):
        self._datahandler = self._datahandlers_registry.get_datahandler(run_config.dataset())
        self._label2idx, self._idx2label = self._datahandler.label2idx()
        self._num_labels = len(self._label2idx)
        base_model_class = self._models_registry.get_model(run_config.model_type())
        self._base_model = base_model_class(run_config, self._num_labels)
        self._base_model.to(device)
        self._config = Config.instance()
        self._task_criterion = nn.CrossEntropyLoss().to(device)


        self._base_model_optimizer = Adam(
            self._base_model.parameters(),
            lr=LEARNING_RATE,
        )

        self.losses = []
        self.task_losses = []
        self._total_count = 0
        self._answer_count = 0
        self._test_data = self._datahandler.test_data()
        self._train_data = self._datahandler.train_data()
        self._featurized_context_cache = {}

    def train(self, run_config):
        self._jade_logger.new_experiment()
        self._jade_logger.set_experiment_type('classification')
        self._jade_logger.set_total_epochs(run_config.epochs())
        random.seed(22)
        random.shuffle(self._train_data)
        for epoch_i in range(1, run_config.epochs()):
            self._jade_logger.new_epoch()
            self._train_epoch(epoch_i, run_config)
            self._eval_epoch(epoch_i, run_config)
            
    def _train_epoch(self, epoch_i, run_config):
        jadelogger_epoch = self._jade_logger.current_epoch()
        train_data = self._train_data
        data_size = len(train_data)
        jadelogger_epoch.set_size(data_size)
        self._jade_logger.new_train_batch()
        for datum_i, datum in enumerate(train_data):
            if datum_i % 100 == 0:
                print('train', datum_i)
            self._train_datum(run_config, datum_i, datum)

    def _eval_epoch(self, epoch_i, run_config):
        test_data = self._test_data
        self._jade_logger.new_evaluate_batch()
        self._labels = []
        for datum_i, datum in enumerate(test_data):
            if datum_i % 100 == 0:
                print('eval', datum_i)

            self._infer_datum(datum, run_config)
        f1 = self.calculate_f1(self._labels)
        print(f1)

    def _train_datum(self, run_config, datum_i, classification_datum: ClassificationDatum):
        text = classification_datum.text()
        if len(text.split()) > 300:
            return
        wordid2tokenid, tokens = self._base_model.wordid2tokenid(text)
        sentence_classification_output = self._base_model(text)
        if sentence_classification_output is None:
            return
        bitmaps = self._datum2bitmap(classification_datum, tokens, run_config)
        answer = []
        answer_bitmap = torch.Tensor(bitmaps['answer_bitmap']).to(device)
        answer_bitmap = torch.Tensor(answer_bitmap).to(device).unsqueeze(0)
        loss = self._task_criterion(sentence_classification_output, answer_bitmap)
        answer_index = torch.argmax(sentence_classification_output).item()
        self._jade_logger.new_train_datapoint(bitmaps['required_answer'], answer, loss.item(), {"text": text})
        self.losses += [loss]
        if len(self.losses) >= BATCH_SIZE:
            (sum(self.losses)/BATCH_SIZE).backward()
            self._base_model_optimizer.step()
            self._base_model.zero_grad()
            self.losses = []
            self._jade_logger.new_train_batch()

    def _datum2bitmap(self, classification_datum, tokens, run_config):
        answer = classification_datum.label()
        answer_bitmap = [0] * self._num_labels
        if isinstance(answer, list):
            for answer_i in answer:
                answer_bitmap[self._label2idx.get(answer_i)] = 1
        else:
            answer_bitmap[self._label2idx.get(answer)] = 1
        bitmaps = {
            'required_answer': answer,
            'answer_bitmap': answer_bitmap,
        }
        return bitmaps

    def _infer_datum(self, classification_datum: ClassificationDatum, run_config):
        text = classification_datum.text()
        if len(text.split()) > 300:
            return
        with torch.no_grad():
            sentence_classification_output = self._base_model(text)
            if sentence_classification_output is None:
                return
            wordid2tokenid, tokens = self._base_model.wordid2tokenid(text)
            bitmaps = self._datum2bitmap(classification_datum, tokens, run_config)
            answer_bitmap = bitmaps['answer_bitmap']
            losses = []
            answer_tensor = torch.Tensor(answer_bitmap).to(device).unsqueeze(0)
            loss = self._task_criterion(sentence_classification_output, answer_tensor)
            losses.append(loss)
            answer_index = torch.argmax(sentence_classification_output).item()
            answer = self._idx2label[answer_index]
            self._labels.append((bitmaps['required_answer'], answer))
            self._jade_logger.new_evaluate_datapoint(bitmaps['required_answer'], answer, loss.item(), {"text": text})

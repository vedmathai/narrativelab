import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from transformers import BigBirdForQuestionAnswering, BigBirdTokenizer
from collections import defaultdict
from jadelogs import JadeLogger
import random


from factuality.common.config import Config
from factuality.tasks.agreement.datamodels.agreement_datum import AgreementDatum
from factuality.tasks.agreement.datahandlers.datahandlers_registry import DatahandlersRegistry
from factuality.tasks.agreement.models.registry import ModelsRegistry


LEARNING_RATE = 1e-5
BATCH_SIZE = 24
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

answer_dict = {
    "0": 0,
    "1": 1,
    "2": 2,
}

answer_reverse_dict = {
    0: "disagree",
    1: "neutral",
    2: "agree",
}


class AgreementClassificationTrainBase:
    def __init__(self):
        self._jade_logger = JadeLogger()
        self._datahandlers_registry = DatahandlersRegistry()
        self._models_registry = ModelsRegistry()

    def load(self, run_config):
        self._datahandler = self._datahandlers_registry.get_datahandler(run_config.dataset())
        base_model_class = self._models_registry.get_model(run_config.model_type())
        self._base_model = base_model_class(run_config)
        self._base_model.to(device)
        self._config = Config.instance()
        self._task_criterion = nn.CrossEntropyLoss().to(device)


        self._base_model_optimizer = Adam(
            self._base_model.parameters(),
            lr=LEARNING_RATE,
        )

        self._total_count = 0
        self._answer_count = 0
        self._test_data = self._datahandler.test_data().data()
        self._train_data = self._datahandler.train_data().data()
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
        batch = []
        for datum_i, datum in enumerate(train_data):
            print(datum_i)
            if datum_i % BATCH_SIZE == 0 and len(batch) != 0:
                self._train_batch(run_config, batch)
                batch = []
            else:
                batch.append(datum)

    def _eval_epoch(self, epoch_i, run_config):
        test_data = self._test_data
        self._jade_logger.new_evaluate_batch()
        self._labels = []
        batch = []
        for datum_i, datum in enumerate(test_data):
            if len(batch) % BATCH_SIZE == 0 and len(batch) != 0:
                self._eval_batch(run_config, batch)         
                batch = []
            else:
                batch.append(datum)
        f1 = self.calculate_f1(self._labels)
        print(f1)

    def _train_batch(self, run_config, batch):
        sentence_classification_outputs = self._base_model(batch)
        losses = []
        for agreement_datum_i, agreement_datum in enumerate(batch):
            text_1 = agreement_datum.text_1()
            text_2 = agreement_datum.text_2()
            sentence_classification_output = sentence_classification_outputs[agreement_datum_i]
            if sentence_classification_output is None:
                continue
            answer_bitmap = [[0, 0, 0] for _ in sentence_classification_output]
            bitmaps = self._datum2bitmap(agreement_datum, run_config)
            answer = []
            answer_bitmap = torch.Tensor(bitmaps['answer_bitmap']).to(device)
            answer_bitmap = torch.Tensor(answer_bitmap).to(device).unsqueeze(0)
            loss = self._task_criterion(sentence_classification_output, answer_bitmap)
            answer_index = torch.argmax(sentence_classification_output).item()
            answer = answer_reverse_dict[answer_index]
            self._jade_logger.new_train_datapoint(bitmaps['required_answer'], answer, loss.item(), {"text_1": text_1, "text_2": text_2})
            losses += [loss]
        (sum(losses)/BATCH_SIZE).backward()
        self._base_model_optimizer.step()
        self._base_model.zero_grad()
        self._jade_logger.new_train_batch()

    def _datum2bitmap(self, quote_datum, run_config):
        answer = quote_datum.label()
        answer_bitmap = [0] * 3
        answer_bitmap[answer_dict.get(answer)] = 1
        answer = answer_reverse_dict[int(answer)]
        bitmaps = {
            'required_answer': answer,
            'answer_bitmap': answer_bitmap,
        }
        return bitmaps

    def _eval_batch(self, run_config, batch):

        with torch.no_grad():
            sentence_classification_outputs = self._base_model(batch)
            for agreement_datum_i, agreement_datum in enumerate(batch):
                text_1 = agreement_datum.text_1()
                text_2 = agreement_datum.text_2()
                sentence_classification_output = sentence_classification_outputs[agreement_datum_i]
                if sentence_classification_output is None:
                    continue
                bitmaps = self._datum2bitmap(agreement_datum, run_config)
                answer_bitmap = bitmaps['answer_bitmap']
                answer_tensor = torch.Tensor(answer_bitmap).to(device).unsqueeze(0)
                loss = self._task_criterion(sentence_classification_output, answer_tensor)
                answer_index = torch.argmax(sentence_classification_output).item()
                answer = answer_reverse_dict[answer_index]
                self._labels.append((bitmaps['required_answer'], answer))
                self._jade_logger.new_evaluate_datapoint(bitmaps['required_answer'], answer, loss.item(), {"text_1": text_1, "text_2": text_2})

    def calculate_f1(self, labels):
        precision_list = defaultdict(list)
        recall_list = defaultdict(list)
        precision = defaultdict(int)
        recall = defaultdict(int)
        f1 = defaultdict(list)
        for l in labels:
            if l[0] == l[1]:
                precision_list[l[0]] += [1]
                recall_list[l[1]] += [1]
            else:
                recall_list[l[1]] += [0]
                precision_list[l[0]] += [0]
        for key in precision_list:
            if len(precision_list[key]) > 0:
                precision[key] = sum(precision_list[key]) / len(precision_list[key])
            if len(recall_list[key]) > 0:
                recall[key] = sum(recall_list[key]) / len(recall_list[key])
            if precision[key] + recall[key] > 0:
                f1[key] = (2 * precision[key] * recall[key]) / (precision[key] + recall[key])
        f1_mean = np.mean(list(f1.values()))
        return f1_mean
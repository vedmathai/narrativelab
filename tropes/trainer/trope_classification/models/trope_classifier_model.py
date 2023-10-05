import os
import torch.nn as nn
from torch import cat
import torch
from transformers import RobertaModel, RobertaTokenizer

from tropes.common.config.config import Config

LLM_INPUT = 768

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TropeClassifierModel(nn.Module):
    def __init__(self, run_config, num_labels, dropout=0.5):
        super(TropeClassifierModel, self).__init__()
        config = Config.instance()
        self._experiment_type = config.experiment_type()
        self._save_location = config.model_save_location()
        self._llm = RobertaModel.from_pretrained('roberta-base').to(device) # noqa
        self._tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self._dropout = nn.Dropout(dropout)
        self._linear_1 = nn.Linear(LLM_INPUT, 352).to(device)
        self._relu = nn.ReLU()
        self._classifier = nn.Linear(352, num_labels).to(device)

    def forward(self, datum):
        sentence = datum.sentence()
        inputs = self._tokenizer([sentence], return_tensors="pt", padding='longest', max_length=512, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self._llm(**inputs)
        pooler_output = outputs.last_hidden_state[0][0].unsqueeze(0)
        dropout_output = self._dropout(pooler_output)
        base_classification_output = self._linear_1(dropout_output)
        activation_output = self._relu(base_classification_output)
        output = self._classifier(activation_output)
        return output

    def save(self):
        state_dict = self.state_dict()
        torch.save(state_dict, self._save_location)

    def load(self):
        if os.path.exists(self._save_location):
            state_dict = torch.load(self._save_location)
            self.load_state_dict(state_dict, strict=False)
        else:
            print('Warning: Model doesn\'t exist. Going with default '
                  'initialized')

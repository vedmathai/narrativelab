import os
import re
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel
import torch

from factuality.common.config import Config

hidden_layer_size = 16

class AgreementClassificationBase(nn.Module):

    def __init__(self, run_config):
        super().__init__()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('AgreementClassificationBase')
        config = Config.instance()
        if run_config.llm() == 'roberta':
            self._tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
            self._model = RobertaModel.from_pretrained('roberta-base')
        modules = [self._model.embeddings, *self._model.encoder.layer[:-2]]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = True
        self._run_config = run_config
        self._dropout = nn.Dropout(0.5).to(self._device)
        self._base_layer_classifier = torch.nn.Linear(768, hidden_layer_size).to(self._device)
        self._base_classifier_activation = nn.Tanh().to(self._device)
        self._classifier = torch.nn.Linear(hidden_layer_size, 3).to(self._device)

    def forward(self, batch):
        token_classification_outputs = []
        for agreement_datum in batch:
            text_1 = agreement_datum.text_1()
            text_2 = agreement_datum.text_2()
            text_1 = self._process_text(text_1)
            text_2 = self._process_text(text_2)
            inputs = self._tokenizer([text_1], [text_2], return_tensors="pt", padding='longest', max_length=1000)
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            outputs = self._model(**inputs)
            pooled_output = outputs.pooler_output
            dropout_output = self._dropout(pooled_output)
            base_classification_output = self._base_layer_classifier(dropout_output)
            activation_output = self._base_classifier_activation(base_classification_output)
            token_classification_output = self._classifier(activation_output)
            token_classification_outputs.append(token_classification_output)
        return token_classification_outputs
    
    def _process_text(self, text):
        text = re.sub("@ @ @ @ @ @ @", '', text)
        text = re.sub("` `", '"', text)
        text = re.sub("' re", 'are', text)
        return text

    def wordid2tokenid(self, text):
        wordid2tokenid = {}
        inputs = self._tokenizer([text], return_tensors="pt", padding='longest', max_length=1000)
        tokens = self._tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]) #input tokens
        stoken_i = 0
        for ti, t in enumerate(tokens):
            if t[0] == '▁':
                wordid2tokenid[stoken_i] = ti
                stoken_i += 1
        return wordid2tokenid, tokens

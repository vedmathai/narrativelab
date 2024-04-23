import os
import re
import torch.nn as nn
from transformers import RobertaTokenizerFast, RobertaModel
import torch

from coreference.common.config.config import Config

hidden_layer_size = 16

class CoreferenceClassifierModel(nn.Module):

    def __init__(self, run_config):
        super().__init__()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = Config.instance()
        self._save_location = config.model_save_location()
        if run_config.llm() == 'roberta':
            self._tokenizer = RobertaTokenizerFast.from_pretrained('transformers_cache/roberta-base', local_files_only=True) 
            self._model = RobertaModel.from_pretrained('roberta-base').to(self._device) # noqa
        modules = [self._model.embeddings, *self._model.encoder.layer[:]]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = True
        self._run_config = run_config
        self._dropout = nn.Dropout(0.5).to(self._device)
        self._base_layer_classifier = torch.nn.Linear(768 * 2, hidden_layer_size).to(self._device)
        self._base_classifier_activation = nn.Tanh().to(self._device)
        self._classifier = torch.nn.Linear(hidden_layer_size, 2).to(self._device)

    def forward(self, datum):
        text = datum.text()
        inputs = self._tokenizer([text], return_tensors="pt", padding='longest', max_length=511, truncation=True)
        tokens = self._tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        # Event 1
        event_1 = datum.event_1()
        event_1_start = datum.event_1_start()
        event_1_end = datum.event_1_end()
        try:
            start_char = inputs.word_to_chars(event_1_start)
            end_char = inputs.word_to_chars(event_1_end)
        except:
            return torch.tensor([0.0, 0.0], requires_grad=True).to(self._device)
        start_token = inputs.char_to_token(start_char.start)
        end_token = inputs.char_to_token(end_char.start)
        if datum.label() == 1:
            offset = self.further_offset(tokens[start_token-20: end_token+20], event_1)
            event_1_start = start_token + offset - 20
            event_1_end = end_token + offset - 19

        # Event 2
        event_2 = datum.event_2()
        event_2_start = datum.event_2_start()
        event_2_end = datum.event_2_end()
        try:
            start_char = inputs.word_to_chars(event_2_start)
            end_char = inputs.word_to_chars(event_2_end)
        except:
            return torch.tensor([0.0, 0.0], requires_grad=True).to(self._device)
        start_token = inputs.char_to_token(start_char.start)
        end_token = inputs.char_to_token(end_char.start)
        offset = self.further_offset(tokens[start_token-20: end_token+20], event_2)
        event_2_start = start_token + offset - 20
        event_2_end = end_token + offset - 19

        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        outputs = self._model(**inputs)
        output_1 = outputs.last_hidden_state[0][event_1_start]
        output_2 = outputs.last_hidden_state[0][event_2_start]
        pooled_output = torch.cat((output_1, output_2), 0)
        dropout_output = self._dropout(pooled_output)
        base_classification_output = self._base_layer_classifier(dropout_output)
        activation_output = self._base_classifier_activation(base_classification_output)
        token_classification_output = self._classifier(activation_output)
        return token_classification_output

    def further_offset(self, tokens, event):
        first_word = event.split(' ')[0]
        current_set = []
        offset = 0
        for tokeni, token in enumerate(tokens):
            if token[0] != 'Ġ':
                current_set.append(token[1:])
            else:
                st = ''.join(current_set)
                if st == first_word:
                    return current_start
                current_set = [token[1:]]
                current_start = tokeni
        return offset    
    
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

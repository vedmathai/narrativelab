import os
import re
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel
from torch_geometric.nn import GCNConv, GATv2Conv
import torch
import torch.nn.functional as F

from factuality.common.config import Config
from factuality.tasks.quotes.graph_featurizers.narrative_graph_featurizer import NarrativeGraphFeaturizer


hidden_layer_size = 16
number_of_relationships = 17


class AgreementClassificationGraph(nn.Module):

    def __init__(self, run_config):
        super().__init__()
        config = Config.instance()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('AgreementClassificationGraph')
        if run_config.llm() == 'roberta':
            self._tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
            self._model = RobertaModel.from_pretrained('roberta-base').to(self._device)
        modules = [self._model.embeddings, *self._model.encoder.layer[:-2]]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = True
        self._run_config = run_config
        self._dropout = nn.Dropout(0.5).to(self._device)
        self._gat1 = GATv2Conv(768 + number_of_relationships, 768 + number_of_relationships, heads=1).to(self._device)
        self._gat2 = GATv2Conv(16*4, 16 * 4, heads=1).to(self._device)
        self._base_layer_classifier = torch.nn.Linear(((768 + number_of_relationships) * 2 + 768), hidden_layer_size).to(self._device)
        self._base_classifier_activation = nn.Tanh().to(self._device)
        self._classifier = torch.nn.Linear(hidden_layer_size, 3).to(self._device)
        self._featurizer = NarrativeGraphFeaturizer()
        self._featurizer.load(self._device)

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

            features = self._featurizer.featurize(text_1)
            if features is None:
                token_classification_outputs.append(None)
                continue
            x, edge_index, main_narrative_index = features
            h = x
            for i in range(3):
                h = self._gat1(h, edge_index)
                h = F.elu(h)
                h = F.dropout(h, p=0.6, training=self.training)
            dropout_output = self._dropout(h)
            dropout_output_1 = dropout_output[main_narrative_index].unsqueeze(0)
            #h = self._gat2(h, edge_index)

            features = self._featurizer.featurize(text_2)
            if features is None:
                token_classification_outputs.append(None)
                continue
            x, edge_index, main_narrative_index = features
            h = x
            for i in range(3):
                h = self._gat1(h, edge_index)
                h = F.elu(h)
                h = F.dropout(h, p=0.6, training=self.training)
            h2 = h
            dropout_output = self._dropout(h)
            dropout_output_2 = dropout_output[main_narrative_index].unsqueeze(0)
            #h = self._gat2(h, edge_index)
            catted_output = torch.cat([dropout_output_1, dropout_output_2, pooled_output], 1)
            base_classification_output = self._base_layer_classifier(catted_output)
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

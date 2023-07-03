import os
import re
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel
from torch_geometric.nn import GCNConv, GATv2Conv
import torch
import torch.nn.functional as F

from factuality.common.config import Config
from factuality.tasks.quotes.graph_featurizers.narrative_graph_featurizer import NarrativeGraphFeaturizer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_layer_size = 16
number_of_relationships = 17

class QuoteClassificationGraph(nn.Module):

    def __init__(self, run_config):
        super().__init__()
        config = Config.instance()
        if run_config.llm() == 'roberta':
            self._tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
            self._model = RobertaModel.from_pretrained('roberta-base').to(device)
        modules = [self._model.embeddings, *self._model.encoder.layer[:-2]]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = True
        self._run_config = run_config
        self._dropout = nn.Dropout(0.5).to(device)
        self._gat1 = GATv2Conv(768 + number_of_relationships, 768 + number_of_relationships, heads=1).to(device)
        self._gat2 = GATv2Conv(16*4, 16 * 4, heads=1).to(device)
        self._base_layer_classifier = torch.nn.Linear(768 + number_of_relationships + 768, hidden_layer_size).to(device)
        self._base_classifier_activation = nn.Tanh().to(device)
        self._classifier = torch.nn.Linear(hidden_layer_size, 4).to(device)
        self._featurizer = NarrativeGraphFeaturizer()
        self._featurizer.load()

    def forward(self, text):
        text = re.sub("@ @ @ @ @ @ @", '', text)
        text = re.sub("` `", '"', text)
        text = re.sub("' re", 'are', text)
        inputs = self._tokenizer([text], return_tensors="pt", padding='longest', max_length=1000)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self._model(**inputs)
        pooled_output = outputs.pooler_output


        features = self._featurizer.featurize(text)
        if features is None:
            return
        x, edge_index, main_narrative_index = features
        h = x
        for i in range(5):
            h = self._gat1(h, edge_index)
            h = F.elu(h)
            h = F.dropout(h, p=0.6, training=self.training)
        #h = self._gat2(h, edge_index)


        dropout_output = self._dropout(h)
        dropout_output = dropout_output[main_narrative_index].unsqueeze(0)
        catted_output = torch.cat([dropout_output, pooled_output], 1)
        base_classification_output = self._base_layer_classifier(catted_output)
        activation_output = self._base_classifier_activation(base_classification_output)
        token_classification_output = self._classifier(activation_output)
        return token_classification_output
    
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
